'''STEP 2 in the dose-response pipeline.

Fits a time-evolving MAP estimate model of the negative control distribution.
This is an empirical Bayes approach to borrow statistical strength from the
other plates scanned at nearby dates; especially useful if step 1 removed a
large portion of the negative control wells.

See element 864 in the GDSC data for a good example of why this estimation
approach is better than the original one that simply takes a mean.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
import numpy as np
import pandas as pd
import sys
import seaborn as sns
from density_filtering import density_regression, density_regression_cv, predict
from dateutil.parser import parse as dateparse
from utils import cols_as_np, isnumeric

def load_dataset(infile):
    # Handle reading in xlsx or csv files
    if infile.endswith('xlsx'):
        return pd.read_excel(infile, header=0)
    return pd.read_csv(infile, parse_dates=['PLATE_DATE'], infer_datetime_format=True, header=0)

if __name__ == '__main__':
    # Load the dataset
    df = load_dataset(sys.argv[1])

    # Select only the unique scans
    df_scans = df.groupby('SCAN_ID').first().reset_index()

    # Total number of each well type
    npos = 48
    nneg = 32
    ndose = 9

    # Get the names of the negative and positive control columns
    neg_cols = ['blank{}'.format(i) for i in range(1,nneg+1)]
    pos_cols = ['control{}'.format(c) for c in range(1,npos+1)]
    treatment_cols = ['raw_max'] + ['raw{}'.format(i) for i in range(2,ndose+1)]

    # Split everything into the four site/assay combinations
    batch_1a = df_scans[(df_scans['DRUG_ID'] > 1000) & (df_scans['ASSAY'] == 'a')] # 1a
    batch_1s = df_scans[(df_scans['DRUG_ID'] > 1000) & (df_scans['ASSAY'] == 's')] # 1s
    batch_2a = df_scans[(df_scans['DRUG_ID'] <= 1000) & (df_scans['ASSAY'] == 'a')] # 2a
    batch_2s = df_scans[(df_scans['DRUG_ID'] <= 1000) & (df_scans['ASSAY'] == 's')] # 2s
    batches = [batch_1a, batch_1s, batch_2a, batch_2s]
    batch_names = ['Site 1, Assay A', 'Site 1, Assay S', 'Site 2, Assay A', 'Site 2, Assay S']

    # plt.close()
    # plt.clf()
    # fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2)
    # axes = [ax1, ax2, ax3, ax4]
    # colors = ['blue', 'orange', 'green', 'purple']
    batch_fits = []
    batch_cols = {'SCAN_ID': [],
                  'Neg_Prior_Mean': [],
                  'Neg_Prior_Variance': [],
                  'Neg_Prior_Shape': [],
                  'Neg_Prior_Rate': [],
                  'Neg_Posterior_Shape': [],
                  'Neg_Posterior_Rate': [],
                  'Neg_MAP_Estimate': []}
    all_lams = []
    for batch_idx, (batch, name) in enumerate(zip(batches, batch_names)):
        dates = batch['PLATE_DATE'].apply(lambda x: x.toordinal())
        
        # Negative controls
        neg_ctrls = batch[neg_cols].values
        neg_ctrls[~np.frompyfunc(isnumeric, 1,1)(neg_ctrls).astype(bool)] = np.nan
        neg_ctrls = neg_ctrls.astype(float)
        neg_meds = np.nanmedian(neg_ctrls, axis=1) #### Aggregation procedure
        
        # Fit the model via cross-validation
        x = dates[~np.isnan(neg_meds)]
        y = neg_meds[~np.isnan(neg_meds)]
        fit, lams = density_regression_cv(x, y, nlam1=10, nlam2=10, nfolds=5)
        fit_x, means, variances = fit
        batch_fits.append(fit)
        all_lams.append(lams)

        # Get the predicted prior mean and variance for every scan
        pred_mean, pred_var = predict(x, fit)

        # Convert from mean/var to gamma shape/rate parameterization
        shapes, rates = means**2 / variances, means / variances
        prior_shape, prior_rate = pred_mean**2 / pred_var, pred_mean / pred_var

        # Calculate the posterior predictions under poisson likelihood assumption
        # where we filter out random contamination by taking the middle 50% of the data
        middle_neg_ctrls = np.copy(neg_ctrls)
        middle_neg_ctrls[(neg_ctrls < np.nanpercentile(neg_ctrls, 25)) | (neg_ctrls > np.nanpercentile(neg_ctrls, 75))] = np.nan
        post_shape, post_rate = prior_shape + np.nansum(middle_neg_ctrls, axis=1), prior_rate + (~np.isnan(middle_neg_ctrls)).sum(axis=1)
        map_lambda = post_shape / post_rate

        # Add everything to the dictionary
        batch_cols['SCAN_ID'].extend(batch['SCAN_ID'])
        batch_cols['Neg_Prior_Mean'].extend(pred_mean)
        batch_cols['Neg_Prior_Variance'].extend(pred_var)
        batch_cols['Neg_Prior_Shape'].extend(prior_shape)
        batch_cols['Neg_Prior_Rate'].extend(prior_rate)
        batch_cols['Neg_Posterior_Shape'].extend(post_shape)
        batch_cols['Neg_Posterior_Rate'].extend(post_rate)
        batch_cols['Neg_MAP_Estimate'].extend(map_lambda)

        # Pretty plotting of the fits
        with sns.axes_style('white', {'legend.frameon': True}):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=3)
            plt.rc('lines', lw=3)
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

            upper = gamma.ppf(0.95, shapes, scale=1/rates)
            lower = gamma.ppf(0.05, shapes, scale=1/rates)

            plt.plot(x - fit_x.min() + 1, y, 'o', color='gray', label='Observed medians', alpha=0.4)
            plt.plot(fit_x - fit_x.min() + 1, means, label='Density estimate', color='orange', zorder=10)
            plt.fill_between(fit_x - fit_x.min() + 1, lower, upper, color='orange', alpha=0.5, zorder=9)

            if batch_idx == 0:
                legend_props = {'weight': 'bold', 'size': 14}
                plt.legend(loc='upper left', prop=legend_props)
            plt.xlabel('Experiment day', fontsize=18, weight='bold')
            plt.ylabel('Negative control', fontsize=18, weight='bold')

            plt.savefig('plots/negative-control-fit-{}.pdf'.format(batch_idx), bbox_inches='tight')
            plt.close()


        # ax.scatter(x, y, alpha=0.4, color='blue')
        # ax.plot(fit_x, means, label='Truth', color='orange')
        # ax.fill_between(fit_x, means + 2*np.sqrt(variances), means - 2*np.sqrt(variances), color='orange', alpha=0.7)
    # plt.savefig('plots/control-prior-fits.pdf', bbox_inches='tight')
    # plt.clf()
    # plt.close()

    print('\nLambdas selected:')
    for name, lams in zip(batch_names, all_lams):
        print('{}: {}'.format(name, lams))
    
    # Save the predictions back to the file
    batch_df = pd.DataFrame(batch_cols)
    batch_df.to_csv('data/control_prior_fits.csv', index=False)

    # Apply the MAP estimate and save the results
    print('Merging')
    df = df.merge(batch_df, on='SCAN_ID')

    # print('Subtracting MAP estimate')
    # df[pos_cols + treatment_cols] = (cols_as_np(df, pos_cols+treatment_cols) - df['Neg_MAP_Estimate'].as_matrix()[:,np.newaxis])#.clip(0, np.inf)

    print('Saving')
    df.to_csv(sys.argv[1].replace('_step1', '_step2'), index=False)


