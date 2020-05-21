'''STEP 3 in the dose-response pipeline.

Estimate the positive control hyper parameters by maximum likelihood
estimation.
'''
import numpy as np
import pandas as pd
import sys
from scipy.special import gammaln, gammaincc
from scipy.stats import gamma, poisson
from scipy.optimize import minimize
from utils import load_dataset, ilogit, cols_as_np


def inc_gamma_loss(logr_logitp, x, c):
    logr, logitp = logr_logitp
    r = np.exp(logr)
    p = ilogit(logitp)
    f = r*np.log((1-p)) + x.sum()*np.log(p) - gammaln(r) + np.log(gammaincc(r+x, c/p)).sum() # gammaln(r + x) + gamma.logcdf(c/p, r+x)
    print(r,p, x, c, f)
    print(np.log(gammaincc(r+x, c/p)))

def approx_loss(ab, x, c):
    a, b = ab
    lam_grid = gamma.ppf(np.linspace(1e-6, 1-1e-6, 1000), a, scale=b)
    weights = gamma.pdf(lam_grid, a, scale=b) / max(1e-20,gamma.pdf(lam_grid, a, scale=b).sum())
    return -np.log((poisson.pmf(x[:,np.newaxis], lam_grid[np.newaxis,:]+c) * weights).sum(axis=1).clip(1e-6,np.inf)).sum()

def fit_gamma_hyperparameters(x, c, nrestarts=10):
    # Quick estimation of the mean and variance to initialize
    x_mean_true = x.mean()
    x_std = x.std()
    best_score, best_params = None, None
    for trial in range(nrestarts):
        x_mean = np.random.normal(x_mean_true, x_std/2)
        # theta0 = np.array([np.log(x_mean**2 / x_std**2), np.log(x_std**2 / x_mean)])
        theta0 = np.array([x_mean**2 / x_std**2, x_std**2 / x_mean])
        print('\t{} Initial: '.format(trial), theta0)
        results = minimize(approx_loss,
                            theta0,
                            args=(x,c),
                            bounds=((0.01,1e5), (0.01,1e8)),
                            method='SLSQP')
        print('\t{} Fit: '.format(trial), results.x, results.fun)
        if np.isnan(results.fun):
            continue
        if best_score is None or results.fun > best_score:
            best_score = results.fun
            # best_params = np.exp(results.x)
            best_params = results.x
    return best_params, best_score

def test_fit():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    a, b, c = 5e1, 1e4, 1e3 # b is a scale param
    n_pos_ctrl = 40
    lam_x = np.random.gamma(a, b, size=n_pos_ctrl)
    x = np.random.poisson(lam_x + c)
    results = fit_gamma_hyperparameters(x, c, nrestarts=10)
    print('Results:', results)
    print('Truth: ', a, b)
    vals, bins, patches = plt.hist(x, bins=20, density=True)
    sns.kdeplot(np.random.poisson(np.random.gamma(a,b,size=100000)+c), label='Truth')
    sns.kdeplot(np.random.poisson(np.random.gamma(results[0][0],results[0][1],size=100000)+c), label='Fit')
    plt.legend(loc='upper left')
    plt.show()

def plot_example(df, row=135865):
    # Pretty plotting of an example fit
    X = df.iloc[row][pos_cols].dropna().values.astype(int)
    a, b, c = df.iloc[row][['Pos_MLE_Shape', 'Pos_MLE_Scale', 'Neg_MAP_Estimate']]
    from scipy.stats import gamma, poisson
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        points = np.round(np.linspace(X.min()-X.std(),X.max()+X.std(),300)).astype(int)
        lam_grid = gamma.ppf(np.linspace(1e-6, 1-1e-6, 1000), a, scale=b)
        weights = gamma.pdf(lam_grid, a, scale=b) / max(1e-20,gamma.pdf(lam_grid, a, scale=b).sum())
        probs = (poisson.pmf(points[:,np.newaxis], lam_grid[np.newaxis,:]+c) * weights).sum(axis=1)
        probs /= probs.sum()

        counts, bins, _ = plt.hist(X, color='gray', label='Observations', alpha=0.4)
        plt.plot(points, probs / probs.max() * counts.max(), label='Fit', color='orange')
        legend_props = {'weight': 'bold', 'size': 14}
        plt.legend(loc='upper left', prop=legend_props)
        plt.xlabel('Measurement value', fontsize=18, weight='bold')
        plt.ylabel('Count', fontsize=18, weight='bold')

        plt.savefig('plots/positive-control-fit.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    # Load the dataset (should be the filtered dataset from step2)
    print('Loading data')
    df = load_dataset(sys.argv[1])
    df_scans = df.groupby('SCAN_ID').first().reset_index()

    # Total number of each well type
    npos = 48
    nneg = 32
    ndose = 9
    nrestarts = 10
    nthreads = 3

    # Get the names of the negative and positive control columns and the treatment columns
    pos_cols = ['control{}'.format(c) for c in range(1,npos+1)]
    neg_map_col = 'Neg_MAP_Estimate'

    fit_cols = {'SCAN_ID': [], 'Pos_MLE_Shape': [], 'Pos_MLE_Scale': []}
    params = []
    for idx,(scan_id, pos_ctrls, neg_map) in enumerate(zip(df_scans['SCAN_ID'],
                                                           cols_as_np(df_scans,pos_cols),
                                                           df_scans[neg_map_col].values)):
        print(idx)

        # Filter out missing controls and model as raw counts
        pos_ctrls = pos_ctrls[~np.isnan(pos_ctrls)].astype(int)

        # Add the parameters
        params.append([pos_ctrls, neg_map, nrestarts])
        fit_cols['SCAN_ID'].append(scan_id)

        (a,b), score = fit_gamma_hyperparameters(pos_ctrls, neg_map)

        fit_cols['Pos_MLE_Shape'].append(a)
        fit_cols['Pos_MLE_Scale'].append(b)

    # Add the columns to the data frame
    print('Merging dataframes')
    fit_df = pd.DataFrame(fit_cols)
    df = df.merge(fit_df, on='SCAN_ID')

    print('Saving')
    df.to_csv(sys.argv[1].replace('_step2', '_step3'), index=False)



