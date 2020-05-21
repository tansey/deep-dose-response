'''STEP 1 in the dose-response pipeline.

Detects and removes potential cross-contaminated negative control wells by
measuring the correlation between negative control well-specific z-scores and
positive control well-specific z-scores. Any negative controls that have a 
pearson correlation above a certain threshold (e.g. rho > 0.15) are considered
contaminated and purged from the dataset.'''
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from utils import isnumeric, pretty_str

def load_dataset(infile):
    # Handle reading in xlsx or csv files
    if infile.endswith('xlsx'):
        return pd.read_excel(infile, header=0)
    return pd.read_csv(infile, header=0)

def get_standardized_controls(batch, cols):
    ctrls = batch[cols].values
    ctrls[~np.frompyfunc(isnumeric, 1,1)(ctrls).astype(bool)] = np.nan
    ctrls = ctrls.astype(float)
    ctrls -= np.nanmean(ctrls, axis=1)[:,np.newaxis]
    ctrls /= np.nanstd(ctrls, axis=1)[:,np.newaxis]
    return ctrls

if __name__ == '__main__':
    # Load the dataset
    print('Loading data (may take a while if it is a large xlsx file)')
    df = load_dataset(sys.argv[1])

    # Select only the unique scans
    df_scans = df.groupby('SCAN_ID').first().reset_index()

    # Total number of each well type
    npos = 48
    nneg = 32

    # Pearson correlation threshold for discarding a well
    corr_threshold = 0.15

    # Get the names of the negative and positive control columns
    neg_cols = ['blank{}'.format(i) for i in range(1,nneg+1)]
    pos_cols = ['control{}'.format(c) for c in range(1,npos+1)]

    # Split everything into the four site/assay combinations
    batch_1a = df_scans[(df_scans['DRUG_ID'] > 1000) & (df_scans['ASSAY'] == 'a')] # 1a
    batch_1s = df_scans[(df_scans['DRUG_ID'] > 1000) & (df_scans['ASSAY'] == 's')] # 1s
    batch_2a = df_scans[(df_scans['DRUG_ID'] <= 1000) & (df_scans['ASSAY'] == 'a')] # 2a
    batch_2s = df_scans[(df_scans['DRUG_ID'] <= 1000) & (df_scans['ASSAY'] == 's')] # 2s
    batches = [batch_1a, batch_1s, batch_2a, batch_2s]
    batch_names = ['Site 1, Assay A', 'Site 1, Assay S', 'Site 2, Assay A', 'Site 2, Assay S']

    # Plot the cross-correlation of all the controls, separated by site/assay
    # Throw out negative controls that are too correlated with the positive controls
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = [ax1, ax2, ax3, ax4]
    uncorrelated_neg_cols = []
    for batch, name, ax in zip(batches, batch_names, axes):
        print('\n{}'.format(name))
        print(batch.describe())
        neg_ctrls = get_standardized_controls(batch, neg_cols)
        pos_ctrls = get_standardized_controls(batch, pos_cols)

        # # Shuffle controls around to make it look a little more interpretable
        # if 'Site 1' in name or 'Assay A' in name:
        #     temp = np.copy(neg_ctrls)
        #     temp[:,:(temp.shape[1]/2)] = neg_ctrls[:,::2]
        #     temp[:,(temp.shape[1]/2):] = neg_ctrls[:,1::2]
        #     neg_ctrls = temp
        # # Site 2 seems to have their positive controls arranged differently
        # nctrls = pos_ctrls.shape[1]
        # tempctrls = np.copy(pos_ctrls)
        # tempctrls[:,:(nctrls/3)] = pos_ctrls[:,::3]
        # tempctrls[:,(nctrls/3):(2*nctrls/3)] = pos_ctrls[:,1::3]
        # tempctrls[:,(2*nctrls/3):] = pos_ctrls[:,2::3]
        # pos_ctrls = tempctrls
        # if name == 'Site 2, Assay S':
        #     # temp = np.copy(pos_ctrls)
        #     # temp[:,:(temp.shape[1]/2)] = pos_ctrls[:,::2]
        #     # temp[:,(temp.shape[1]/2):] = pos_ctrls[:,1::2]
        #     # pos_ctrls = temp
        #     temp = np.copy(neg_ctrls)
        #     temp[:,:(temp.shape[1]/4)] = neg_ctrls[:,::4]
        #     temp[:,(temp.shape[1]/4):(temp.shape[1]/2)] = neg_ctrls[:,1::4]
        #     temp[:,(temp.shape[1]/2):(3*temp.shape[1]/4)] = neg_ctrls[:,2::4]
        #     temp[:,(3*temp.shape[1]/4):] = neg_ctrls[:,3::4]
        #     neg_ctrls = temp

        # Combine the two controls into a single dataset and look at the
        # cross-correlations between all wells
        ctrls = np.concatenate((neg_ctrls, pos_ctrls), axis=1)
        c = pd.DataFrame(ctrls).corr().as_matrix()

        # Find the maximum negative-positive correlations
        max_corr = np.nanmax(c[:nneg,nneg:], axis=1)
        print('Maximum pearson correlations for wells:', pretty_str(max_corr))

        # Discard all columns above a threshold
        discarded = [col for corr, col in zip(max_corr, neg_cols) if corr > corr_threshold]
        df.loc[df.SCAN_ID.isin(batch.SCAN_ID), discarded] = np.nan

        print('Total negative wells: {} Discarded: {} Kept: {}'.format(len(max_corr), len(discarded), len(max_corr) - len(discarded)))

        if ax == axes[0]:
            im = sns.heatmap(c, ax=ax, cmap='PiYG', xticklabels=[""], yticklabels=[""], cbar = False, vmin=-0.5, vmax=0.5)
        else:
            sns.heatmap(c, ax=ax, cmap='PiYG', xticklabels=[""], yticklabels=[""], cbar = False, vmin=-0.5, vmax=0.5)
        ax.set_title(name)
    mappable = im.get_children()[0]
    plt.colorbar(mappable, ax = [ax1,ax2,ax3,ax4],orientation = 'horizontal')
    plt.savefig('plots/cross-contamination.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    # Save the results back to file
    df.to_csv(sys.argv[1].replace('.xlsx', '_step1.csv'), index=False)

    print('Done!')


