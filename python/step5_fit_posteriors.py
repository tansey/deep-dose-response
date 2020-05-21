'''
Estimate the dose-response covariance matrix prior on a per-drug basis.
'''
import numpy as np
from gass import gass
from scipy.stats import invwishart, poisson, gamma
from utils import ilogit
from step4_fit_prior_fast import create_predictive_model, NeuralModel, EmpiricalBayesOptimizer

def beta_mcmc(ebo, drug_idx,
                nburn=500, nsamples=1000, nthin=1, **kwargs):
    indices = np.arange(ebo.Y.shape[0])[np.any(ebo.obs_mask[:,drug_idx].astype(bool), axis=1)]

    np.set_printoptions(suppress=True, precision=2)
    print('Drug {} ({}) total samples {}'.format(drug_idx, ebo.drugs[drug_idx], len(indices)))

    # Get the offsets and grids
    lam_grid = ebo.lam_grid[indices, drug_idx]
    weights = gamma.pdf(lam_grid, ebo.A[indices, drug_idx, None], scale=ebo.B[indices, drug_idx, None])#.clip(1e-10, np.inf)
    weights /= weights.sum(axis=-1, keepdims=True)
    Y = ebo.Y[indices, drug_idx]
    C = ebo.C[indices, drug_idx]

    # Get the empirical Bayes predicted mean and back out the logits
    tau_hat = ebo.mu[indices, drug_idx].clip(1e-4, 1-1e-4)
    # tau_hat = ebo.predict_mu(ebo.X[indices])[:,drug_idx].clip(1e-4, 1-1e-4)
    Mu = np.log(tau_hat / (1-tau_hat))

    # Initialize at the simple Mu point
    # Beta = np.copy(Mu)

    # Vague prior on Sigma
    nu = Mu.shape[1]+1
    Psi = np.eye(Mu.shape[1])
    Psi *= nu - Mu.shape[1] + 1

    # Initialize sigma with a simple inverse wishart draw
    # Sigma = invwishart.rvs(nu, Psi)

    # Initialize with the empirical estimates
    Tau_empirical = ((Y - C[...,None]) / lam_grid[...,lam_grid.shape[-1]//2,None]).clip(1e-4,1-1e-4)
    Beta = np.maximum.accumulate(np.log(Tau_empirical / (1-Tau_empirical)), axis=1) + np.cumsum([1e-2]*Y.shape[-1])
    # Sigma = invwishart.rvs(nu+Mu.shape[0], Psi+(Beta - Mu).T.dot(Beta - Mu))
    Sigma = (Beta - Mu).T.dot(Beta - Mu) / Mu.shape[0]
    print(Sigma)

    # Create the results arrays
    Beta_samples = np.zeros((nsamples, Beta.shape[0], Beta.shape[1]))
    Sigma_samples = np.zeros((nsamples, Sigma.shape[0], Sigma.shape[1]))
    Loglikelihood_samples = np.zeros(nsamples)

    # Setup the monotonicity constraint
    C_mono = np.array([np.concatenate([np.zeros(i), [-1,1], np.zeros(Mu.shape[1]-i-2), [0]]) for i in range(Mu.shape[1]-1)])

    # Log-likelihood function for a sigle curve
    def log_likelihood(z, idx):
        expanded = len(z.shape) == 1
        if expanded:
            z = z[None]
        z = z[...,None] # room for lambda grid
        lam = lam_grid[idx,None,None] # room for z grid and multiple doses
        c = C[idx,None,None,None]
        w = weights[idx,None,None]
        y = Y[idx,None,:,None]
        result = np.nansum(np.log((poisson.pmf(y, ilogit(z)*lam + c) * w).clip(1e-10, np.inf).sum(axis=-1)), axis=-1)
        if expanded:
            return result[0]
        return result

    # Initialize betas with draws from the prior
    Cur_log_likelihood = np.zeros(Mu.shape[0])
    chol = np.linalg.cholesky(Sigma)

    # Run the MCMC sampler
    for step in range(nburn+nsamples*nthin):
        print('MCMC step {}'.format(step))

        # Ellipitical slice sample for each beta
        for idx, beta in enumerate(Beta):
            Beta[idx], Cur_log_likelihood[idx] = gass(beta, chol, log_likelihood, C_mono,
                                                        mu=Mu[idx],
                                                        cur_ll=None if step == 0 else Cur_log_likelihood[idx],
                                                        ll_args=idx, chol_factor=True)

        # Conjugate prior update for Sigma
        # Sigma = invwishart.rvs(nu+Mu.shape[0], Psi+(Beta - Mu).T.dot(Beta - Mu))

        # Cholesky representation
        # chol = np.linalg.cholesky(Sigma)

        # Save this sample after burn-in and markov chain thinning
        if step < nburn or ((step-nburn) % nthin) != 0:
            continue

        # Save the samples
        sample_idx = (step - nburn) // nthin
        Beta_samples[sample_idx] = Beta
        Sigma_samples[sample_idx] = Sigma
        Loglikelihood_samples[sample_idx] = Cur_log_likelihood.sum()

        # if sample_idx == 50 or sample_idx == 500 or sample_idx == (nsamples-1):
        #     import matplotlib.pyplot as plt
        #     import seaborn as sns
        #     Tau_unclipped = ((Y - C[...,None]) / lam_grid[...,lam_grid.shape[-1]//2,None])
        #     for idx in range(30):
        #         plt.scatter(np.arange(Y.shape[1])[::-1], Tau_unclipped[idx], color='gray', label='Observed')
        #         plt.plot(np.arange(Y.shape[1])[::-1], ilogit(Mu[idx]), color='orange', label='Prior')
        #         plt.plot(np.arange(Y.shape[1])[::-1], ilogit(Beta_samples[:sample_idx+1,idx].mean(axis=0)), color='blue', label='Posterior')
        #         plt.fill_between(np.arange(Y.shape[1])[::-1],
        #                          ilogit(np.percentile(Beta_samples[:sample_idx+1,idx], 5, axis=0)),
        #                          ilogit(np.percentile(Beta_samples[:sample_idx+1,idx], 95, axis=0)),
        #                          alpha=0.3, color='blue')
        #         plt.legend(loc='lower left')
        #         plt.savefig('../plots/posteriors-drug{}-sample{}.pdf'.format(drug_idx, idx), bbox_inches='tight')
        #         plt.close()

    return indices, Beta_samples, Sigma_samples, Loglikelihood_samples


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Estimate the dose-response covariance matrix on a per-drug basis.')

    # Experiment settings
    parser.add_argument('name', default='gdsc', help='The project name. Will be prepended to plots and saved files.')
    parser.add_argument('--drug', type=int, help='If specified, fits only on a specific drug. This is useful for parallel/distributed training.')
    parser.add_argument('--drug_responses', default='data/raw_step3.csv', help='The dataset file with all of the experiments.')
    parser.add_argument('--genomic_features', default='data/gdsc_all_features.csv', help='The file with the cell line features.')
    parser.add_argument('--drug_details', default='data/gdsc_drug_details.csv', help='The data file with all of the drug information (names, targets, etc).')
    parser.add_argument('--plot_path', default='plots', help='The path where plots will be saved.')
    parser.add_argument('--save_path', default='data', help='The path where data and models will be saved.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--torch_threads', type=int, default=1, help='The number of threads that pytorch can use in a fold.')
    parser.add_argument('--no_fix', action='store_true', default=False, help='Do not correct the dosages.')
    parser.add_argument('--verbose', action='store_true', help='If specified, prints progress to terminal.')
    parser.add_argument('--nburn', type=int, default=500, help='Number of MCMC burn-in steps.')
    parser.add_argument('--nsamples', type=int, default=1500, help='Number of MCMC steps to use.')
    parser.add_argument('--nthin', type=int, default=1, help='Number of MCMC steps between sample steps.')
    
    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Seed the random number generators so we get reproducible results
    np.random.seed(args.seed)
    
    print('Running step 5 with args:')
    print(args)
    print('Working on project: {}'.format(args.name))

    # Create the model directory
    model_save_path = os.path.join(args.save_path, args.name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Load the predictor
    ebo = create_predictive_model(model_save_path, **dargs)
    ebo.load()

    # Fit the posterior on  via MCMC
    drug_idx = args.drug
    indices, Beta, Sigma, loglike = beta_mcmc(ebo, drug_idx, **dargs)


    # Calculate the posterior AUC scores
    Tau = ilogit(Beta)
    AUC = (Tau.sum(axis=2) - 0.5*Tau[:,:,[0,-1]].sum(axis=2)) / (Tau.shape[2]-1)

    posteriors_path = os.path.join(model_save_path, 'posteriors')
    if not os.path.exists(posteriors_path):
        os.makedirs(posteriors_path)
    np.save(os.path.join(posteriors_path, 'betas{}'.format(drug_idx)), Beta)
    np.save(os.path.join(posteriors_path, 'sigmas{}'.format(drug_idx)), Sigma)
    np.save(os.path.join(posteriors_path, 'taus{}'.format(drug_idx)), Tau)
    np.save(os.path.join(posteriors_path, 'aucs{}'.format(drug_idx)), AUC)

    ### Plot examples
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Get the offsets and grids
    lam_grid = ebo.lam_grid[indices, drug_idx]
    weights = gamma.pdf(lam_grid, ebo.A[indices, drug_idx, None], scale=ebo.B[indices, drug_idx, None])#.clip(1e-10, np.inf)
    weights /= weights.sum(axis=-1, keepdims=True)
    Y = ebo.Y[indices, drug_idx]
    C = ebo.C[indices, drug_idx]

    # Get the empirical Bayes predicted mean and back out the logits
    # tau_hat = ebo.mu[indices, drug_idx].clip(1e-4, 1-1e-4)
    tau_hat = ebo.predict_mu(ebo.X[indices])[:,drug_idx].clip(1e-4, 1-1e-4)
    Mu = np.log(tau_hat / (1-tau_hat))

    Tau_unclipped = ((Y - C[...,None]) / lam_grid[...,lam_grid.shape[-1]//2,None])
    # for idx in range(30):
    #     plt.scatter(np.arange(Y.shape[1])[::-1], Tau_unclipped[idx], color='gray', label='Observed')
    #     plt.plot(np.arange(Y.shape[1])[::-1], ilogit(Mu[idx]), color='orange', label='Prior')
    #     plt.plot(np.arange(Y.shape[1])[::-1], ilogit(Beta[:,idx].mean(axis=0)), color='blue', label='Posterior')
    #     plt.fill_between(np.arange(Y.shape[1])[::-1],
    #                      ilogit(np.percentile(Beta[:,idx], 5, axis=0)),
    #                      ilogit(np.percentile(Beta[:,idx], 95, axis=0)),
    #                      alpha=0.3, color='blue')
    #     plt.legend(loc='lower left')
    #     plt.savefig('plots/posteriors-drug{}-sample{}.pdf'.format(drug_idx, idx), bbox_inches='tight')
    #     plt.close()

    

    # Fix bugs -- done and saved
    # df_sanger['DRUG_NAME'] = df_sanger['DRUG_NAME'].str.strip()
    # df_sanger[df_sanger['DRUG_NAME'] == 'BX-795'] = 'BX-796'
    # df_sanger[df_sanger['DRUG_NAME'] == 'SB505124'] = 'SB-505124'
    # df_sanger[df_sanger['DRUG_NAME'] == 'Lestaurtinib'] = 'Lestauritinib'

    # Get all the Sanger-processed AUC scores in a way we can handle it
    sanger_auc_path = os.path.join(args.save_path, 'sanger_auc.npy')
    if not os.path.exists(sanger_auc_path):
        import pandas as pd
        from collections import defaultdict
        df_sanger = pd.read_csv(os.path.join(args.save_path, 'gdsc_auc.csv'), header=0, index_col=0)
        cell_map, drug_map = defaultdict(lambda: -1), defaultdict(lambda: -1)
        for idx, c in enumerate(ebo.cell_lines):
            cell_map[c] = idx
        for idx, d in enumerate(ebo.drugs):
            drug_map[d] = idx
        AUC_sanger = np.full(ebo.Y.shape[:2], np.nan)
        for idx, row in df_sanger.iterrows():
            cidx, didx = cell_map[row['CELL_LINE_NAME']], drug_map[row['DRUG_NAME']]
            if cidx == -1 or didx == -1:
                continue
            AUC_sanger[cidx, didx] = row['AUC']
        np.save(sanger_auc_path, AUC_sanger)
    else:
        AUC_sanger = np.load(sanger_auc_path)
    
    import seaborn as sns
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        overlap = ~np.isnan(AUC_sanger[indices, drug_idx])
        x = AUC_sanger[indices[overlap], drug_idx]
        y = AUC[:,overlap].mean(axis=0)
        plt.scatter(x, y, s=4)
        plt.plot([min(x.min(), y.min()),1], [min(x.min(), y.min()),1], color='red', lw=2)
        plt.xlabel('Original AUC', fontsize=18)
        plt.ylabel('Bayesian AUC', fontsize=18)
        plt.savefig('plots/auc-compare{}.pdf'.format(drug_idx), bbox_inches='tight')
        plt.close()
    











