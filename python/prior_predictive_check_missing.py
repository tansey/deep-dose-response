import numpy as np
from utils import ilogit, cols_as_np

def marginal_log_likelihood(y, mu, Sigma, a, b, c, lam_gridsize=100, prior_samples=1000000):
    print('Marginal log likelihood via (naive) prior sampling strategy with {} prior samples'.format(prior_samples))
    # Use a grid approximation for lambda integral
    from scipy.stats import multivariate_normal as mvn
    beta = np.random.multivariate_normal(mu, Sigma, size=prior_samples)
    tau = ilogit(beta)
    ll = np.array([heldout_log_likelihood(y_i, tau_i, a, b, c, lam_gridsize=lam_gridsize) for y_i, tau_i in zip(y, tau.T)]).T
    beta_probs = mvn.logpdf(beta, mu, Sigma)
    return ll.sum(axis=1) + beta_probs


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pylab as plt
    import sys
    import os
    import argparse
    from utils import load_dataset, pretty_str
    from step4_fit_prior_missing import EmpiricalBayesOptimizer, DrugResponsePrior

    parser = argparse.ArgumentParser(description='Prior predictive error estimation for models trained in step 4.')

    # Experiment settings
    parser.add_argument('model', help='The project name.')
    parser.add_argument('--dataset', default='data/raw_step3.csv', help='The dataset file with all of the experiments.')
    parser.add_argument('--save_path', default='data', help='The path where data and models will be saved.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--fix_dosages', action='store_true', help='Correct the dosages if they are mixed 2x and 4x dilution.')
    parser.add_argument('--nprior_samples', type=int, default=10000, help='The number of samples from the prior to use when calculating the marginal log likelihood.')
    parser.add_argument('--save', action='store_true', help='If specified, saves the results.')
    
    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Seed the random number generators so we get reproducible results
    np.random.seed(args.seed)

    print('Loading response data')
    df = load_dataset(args.dataset) # usually data/raw_step3.csv

    # Get the observations
    ndose = 9
    treatment_cols = ['raw_max'] + ['raw{}'.format(i) for i in range(2,ndose+1)]
    Y = df[treatment_cols].values
    a = df['Pos_MLE_Shape'].values
    b = df['Pos_MLE_Scale'].values
    c = df['Neg_MAP_Estimate'].values

    npos = 48
    pos_cols = ['control{}'.format(c) for c in range(1,npos+1)]
    pos_ctrls = cols_as_np(df, pos_cols)
    pos_std = np.nanstd(pos_ctrls, axis=1)

    # Handle some idiosyncracies of the GDSC dataset
    if args.fix_dosages:
        select = np.any(np.isnan(Y), axis=1)
        Y[select,0::2] = Y[select,:5]
        Y[select,1::2] = np.nan
    else:
        import warnings
        warnings.warn('Fix dosages is not enabled. GDSC data requires fixing; this should only be unspecified on another dataset.')

    print('Loading model from {}'.format(os.path.join(args.save_path, args.model)))
    ebo = EmpiricalBayesOptimizer(restore_path=os.path.join(args.save_path, args.model))
    
    print('Calculating squared error')
    marginal_se = (((Y - a[:,np.newaxis]*b[:,np.newaxis]*ilogit(ebo.prior_mu) - c[:,np.newaxis]) / pos_std[:,np.newaxis])**2)
    mse_dosages = np.nanmean(marginal_se, axis=0)
    mse_curves = np.nanmean(marginal_se, axis=1).mean()

    print('*** Results for model {} ***'.format(args.model))
    print('Individual prior dosage mean squared errors:')
    for dose, mse in enumerate(mse_dosages):
        print('Dosage {}: {:.2f}'.format(dose, mse))
    print('Mean curve MSE: {:.2f}'.format(mse_curves))
    

    if args.save:
        import csv
        with open(os.path.join(os.path.join(args.save_path, args.model), 'prior_predictive.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow([mse_curves] + list(mse_dosages))


