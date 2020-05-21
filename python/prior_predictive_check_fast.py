import numpy as np
from utils import cols_as_np, load_dataset
from step4_fit_prior_fast import create_predictive_model, NeuralModel, EmpiricalBayesOptimizer

def load_pos_mean_std(ebo, drug_responses):
    print('Loading response data')
    df = load_dataset(drug_responses) # usually data/raw_step3.csv

    # Get the positive control standard deviation
    ndose = 9
    npos = 48
    treatment_cols = ['raw_max'] + ['raw{}'.format(i) for i in range(2,ndose+1)]
    pos_cols = ['control{}'.format(c) for c in range(1,npos+1)]
    pos_ctrls = cols_as_np(df, pos_cols)
    pos_mean = np.nanmean(pos_ctrls, axis=1)
    pos_std = np.nanstd(pos_ctrls, axis=1)
    
    D_mean, D_std = np.full(ebo.Y.shape[:2], np.nan), np.full(ebo.Y.shape[:2], np.nan)
    for r,(i,j) in enumerate(ebo.raw_index):
        if i == -1 or j == -1:
            continue
        D_mean[i,j] = pos_mean[r]
        D_std[i,j] = pos_std[r]

    return D_mean, D_std


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
    parser.add_argument('--no_fix', action='store_true', default=False, help='Do not correct the doses.')
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

    # Load the positive control standard deviations
    pos_mean, pos_std = load_pos_mean_std(ebo, args.drug_responses)

    # Filter out missing observations
    ebo.Y[~ebo.obs_mask.astype(bool)] = np.nan

    # import matplotlib.pyplot as plt
    # Z_min_dose = (ebo.Y[...,-1] - pos_mean) / pos_std
    # for i in range(Z_min_dose.shape[0]):
    #     for j in range(Z_min_dose.shape[1]):
    #         if np.isnan(Z_min_dose[i,j]) or np.abs(Z_min_dose[i,j]) < 6:
    #             continue
    #         print('({},{}): {:.2f}'.format(i,j,Z_min_dose[i,j]))
    # Z_min_dose = Z_min_dose[~np.isnan(Z_min_dose)].flatten()
    # print('Biggest outliers:')
    # print(Z_min_dose[np.argsort(np.abs(Z_min_dose))[-20:]])
    # Y_min_dose_clipped = Z_min_dose[np.abs(Z_min_dose) <= 6]
    # print('Without outliers this would be N({},{}^2)'.format(Y_min_dose_clipped.mean(), Y_min_dose_clipped.std()))
    # plt.hist(Z_min_dose, bins=np.linspace(-7,7,1001))
    # plt.savefig('plots/tmp.pdf', bbox_inches='tight')
    # plt.close()

    print('Calculating squared error')
    
    mse = ((ebo.Y - ebo.A[...,None]*ebo.B[...,None]*ebo.mu - ebo.C[...,None]) / pos_std[...,None])**2
    mse_drugs = np.nanmean(mse, axis=0) # Average out the different cell lines
    mse_doses = np.nanmean(mse_drugs, axis=0) # Average out the different dose levels
    mse_curves = np.nanmean(mse_doses).mean() # Average all the drugs

    # print('*** Results for model {} ***'.format(args.name))
    print('Individual prior dose mean squared errors:')
    for dose, mse_dose in enumerate(mse_doses):
        print('Dose {}: {:.2f}'.format(dose, mse_dose))
    print('Mean curve MSE: {:.2f}'.format(mse_curves))
    

    import csv
    with open(os.path.join(model_save_path, 'prior_predictive.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([mse_curves] + list(mse_doses))








