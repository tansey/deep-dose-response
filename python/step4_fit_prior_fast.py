'''STEP 4 fit a deep empirical Bayes prior model via SGD.

Builds an empirical Bayes model to predict the prior over the dose-response
mean-effect curve.

TODO: Generative model description.

We use a neural network to model, trained with stochastic gradient descent.
The features are the mutation, copy number, and gene expression information.
The outputs are the mean and covariance for the MVN prior on the dose-response
in each of the unique drugs.

For numerical purposes, we approximate the double integral by a finite grid
over lambda and Monte Carlo sampling for beta.
'''
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from deep_learning import fit as fit_nn
from utils import load_dataset, batches, create_folds, \
                  pretty_str, ilogit, logsumexp, try_load

class EmpiricalBayesOptimizer:
    def __init__(self, Y=None, # N cell lines x M drugs x T doses
                       A=None, B=None, C=None, # Hyperparameters estimated offline
                       lam_gridsize=100, # Number of discrete points to approximate the NLL integral with
                       restore_path=None,
                       lam_path=None,
                       **kwargs):
        pass

    def setup(self, model_path, cell_lines, drugs, drug_ids, features,
                    X, Y, A, B, C, raw_index,
                    lam_gridsize=100, nfolds=10, **kwargs):
        '''Initializes the model and caches certain statistics.'''
        self.model_path = model_path
        self.cell_lines = cell_lines
        self.drugs = drugs
        self.drug_ids = drug_ids
        self.features = features
        self.X = X
        self.A = A
        self.B = B
        self.C = C
        self.Y = Y
        self.raw_index = raw_index
        
        assert A.shape == Y.shape[:-1]
        assert B.shape == Y.shape[:-1]
        assert C.shape == Y.shape[:-1]

        self.Y_shape = Y.shape
        self.nsamples = Y.shape[0]
        self.ndrugs = Y.shape[1]
        self.ndoses = Y.shape[2]
        self.nfeatures = X.shape[1]

        # Cache which doses are missing and put in dummy values
        from scipy.stats import gamma
        self.obs_mask = (~np.isnan(Y)).astype(int)
        self.A = np.nan_to_num(self.A, nan=1)
        self.B = np.nan_to_num(self.B, nan=1)
        self.C = np.nan_to_num(self.C, nan=1)
        self.Y = np.nan_to_num(self.Y, nan=0)*self.obs_mask + (1-self.obs_mask)*2

        # We approximate the integral over lambda with a finite grid of lam_gridsize points
        print('Caching lambda integral approximation')
        self.lam_gridsize = lam_gridsize
        self.lam_grid = np.transpose(np.linspace(gamma.ppf(1e-3, self.A, scale=self.B),
                                                 gamma.ppf(1-1e-3, self.A, scale=self.B),
                                                 self.lam_gridsize), [1,2,0])
        self.lam_weights = gamma.pdf(self.lam_grid, self.A[...,None], scale=self.B[...,None])
        self.lam_weights = (self.lam_weights / self.lam_weights.sum(axis=-1, keepdims=True)).clip(1e-6, 1-1e-6)
        self.log_lam_weights = np.log(self.lam_weights)
        # np.save(os.path.join(self.model_path, 'lam_grid.npy'), self.lam_grid)
        # np.save(os.path.join(self.model_path, 'lam_weights.npy'), self.lam_weights)

        # Split the data into K folds
        self.nfolds = nfolds
        self.folds = create_folds(self.nsamples, self.nfolds)

        # The out-of-sample predictions
        self.mu = np.full(self.Y.shape, np.nan)

    def loss_fn(self, tidx, tau, train=False):
        #### Calculate the loss as the negative log-likelihood of the data ####
        # Poisson noise model for observations
        rates = tau[...,None] * self.tLamGrid[tidx][:,:,None] + self.tC[tidx][:,:,None,None]
        likelihoods = torch.distributions.Poisson(rates)

        mask = self.tObsMask[tidx]
        # if train:
        #     # Randomly sample a subset of drugs and doses
        #     mask = mask * autograd.Variable(torch.FloatTensor(np.random.random(size=self.tObsMask[tidx].shape) <= 0.5), requires_grad=False)

        # Get log probabilities of the data and filter out the missing observations
        loss = -(logsumexp(likelihoods.log_prob(self.tY[tidx][:,:,:,None])
                           + self.tLogLamWeights[tidx][:,:,None], dim=-1)
                * mask)
        loss = loss.sum(dim=-1) / mask.sum(dim=-1).clamp(1,mask.shape[-1]) # Average all doses in each curve
        loss = loss.sum(dim=-1) / mask.max(dim=-1).values.sum(dim=-1).clamp(1,mask.shape[-2]) # Average across all drugs in each cell line
        # print(loss)
        # loss = -(likelihoods.log_prob(self.tY[tidx][:,:,:,None]) + self.tLogLamWeights[tidx][:,:,None]).exp().sum(dim=-1).clamp(1e-10, 1).sum()
        # loss = -logsumexp(likelihoods.log_prob(self.tY[tidx][:,:,:,None]) + self.tLogLamWeights[tidx][:,:,None], dim=-1).sum()
        return loss


        loss = -(likelihoods.log_prob(self.tY[tidx][:,:,:,None]) + self.tLogLamWeights[tidx][:,:,None]).exp().sum(dim=-1).clamp(1e-10, 1).log()
        print('1', loss)
        print('1 nans: {}'.format(np.isnan(loss.data.numpy()).sum()))
        print('tObsMask', self.tObsMask[tidx])
        print('tObsMask nans: {}'.format(np.isnan(self.tObsMask[tidx].data.numpy()).sum()))
        loss = (loss  * self.tObsMask[tidx]).sum(dim=-1) / self.tObsMask[tidx].sum(dim=-1).clamp(1, np.prod(self.tObsMask[tidx].shape)) # average across available doses
        print('2', loss)
        print('2 nans: {}'.format(np.isnan(loss.data.numpy()).sum()))
        print()
        print('shape', loss.shape)

        loss = loss.sum(dim=1) / self.tObsMask[tidx].max(dim=-1).values.sum(dim=-1)
        print('3', loss)
        print('3 nans: {}'.format(np.isnan(loss.data.numpy()).sum()))
        return loss

    def fit_mu(self, verbose=False, **kwargs):
        self.models = []
        self.train_folds = []
        self.validation_folds = []
        for fold_idx, fold in enumerate(self.folds):
            if verbose:
                print('Fitting model {}'.format(fold_idx), flush=True)

            # Define the training set for this fold
            mask = np.ones(self.X.shape[0], dtype=bool)
            mask[fold] = False

            # Setup some torch variables
            self.tLamGrid = autograd.Variable(torch.FloatTensor(self.lam_grid[mask]), requires_grad=False)
            self.tLamWeights = autograd.Variable(torch.FloatTensor(self.lam_weights[mask]), requires_grad=False)
            self.tLogLamWeights = autograd.Variable(torch.FloatTensor(self.log_lam_weights[mask]), requires_grad=False)
            self.tC = autograd.Variable(torch.FloatTensor(self.C[mask]), requires_grad=False)
            self.tObsMask = autograd.Variable(torch.FloatTensor(self.obs_mask[mask]), requires_grad=False)
            self.tY = autograd.Variable(torch.FloatTensor(self.Y[mask]), requires_grad=False)

            # Fit the model to the data, holding out a subset of cell lines entirely
            results = fit_nn(self.X[mask],
                            lambda: NeuralModel(self.nfeatures, self.ndrugs, self.ndoses, **kwargs),
                            self.loss_fn, verbose=verbose, **kwargs)

            model = results['model']


            # Save the model to file
            torch.save(model, os.path.join(self.model_path, 'model{}.pt'.format(fold_idx)))
            self.models.append(model)
            
            # Save the train, validation, and test folds
            self.train_folds.append(np.arange(self.X.shape[0])[mask][results['train']])
            self.validation_folds.append(np.arange(self.X.shape[0])[mask][results['validation']])
            np.save(os.path.join(self.model_path, 'train_fold{}'.format(fold_idx)), self.train_folds[fold_idx])
            np.save(os.path.join(self.model_path, 'validation_fold{}'.format(fold_idx)), self.validation_folds[fold_idx])
            np.save(os.path.join(self.model_path, 'test_fold{}'.format(fold_idx)), self.folds[fold_idx])

            # Get the out-of-sample predictions
            self.mu[fold] = self.predict_mu(self.X[fold], model_idx=fold_idx)

        # Save the out-of-sample predictions to file
        np.save(os.path.join(self.model_path, 'mu'), self.mu)

    def predict_mu(self, X, model_idx=None):
        if model_idx is None:
            return np.mean([self.predict_mu(X, model_idx=idx) for idx in range(len(self.models))], axis=0)
        return self.models[model_idx].predict(X)

    def _predict_mu_insample(self, fold_idx, sample_idx, drug_idx):
        fold = self.folds[fold_idx]
        mask = np.ones(self.X.shape[0], dtype=bool)
        mask[fold] = False
        tau_hat = self.models[fold_idx].predict(self.X[mask][sample_idx:sample_idx+1])[0,drug_idx]
        tau_empirical = (self.Y[mask][sample_idx,drug_idx] - self.C[mask][sample_idx,drug_idx]) / self.lam_grid[mask][sample_idx,drug_idx][...,self.lam_grid.shape[-1]//2]
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.scatter(np.arange(self.Y.shape[2])[::-1], tau_empirical, color='gray', label='Observed')
        plt.plot(np.arange(self.Y.shape[2])[::-1], tau_hat, color='blue', label='Predicted')
        plt.savefig('plots/mu-insample-fold{}-sample{}-drug{}.pdf'.format(fold_idx, sample_idx, drug_idx), bbox_inches='tight')
        plt.close()

    def _predict_mu_outsample(self, fold_idx, sample_idx, drug_idx):
        fold = self.folds[fold_idx]
        tau_hat = self.models[fold_idx].predict(self.X[fold][sample_idx:sample_idx+1])[0,drug_idx]
        tau_empirical = (self.Y[fold][sample_idx,drug_idx] - self.C[fold][sample_idx,drug_idx]) / self.lam_grid[fold][sample_idx,drug_idx][...,self.lam_grid.shape[-1]//2]
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.scatter(np.arange(self.Y.shape[2])[::-1], tau_empirical, color='gray', label='Observed')
        plt.plot(np.arange(self.Y.shape[2])[::-1], tau_hat, color='blue', label='Predicted')
        plt.savefig('plots/mu-outsample-fold{}-sample{}-drug{}.pdf'.format(fold_idx, sample_idx, drug_idx), bbox_inches='tight')
        plt.close()

    def load(self):
        import warnings
        self.models = []
        self.train_folds = []
        self.validation_folds = []
        for fold_idx, fold in enumerate(self.folds):
            fold_model_path = os.path.join(self.model_path, 'model{}.pt'.format(fold_idx))
            if os.path.exists(fold_model_path):
                self.models.append(torch.load(fold_model_path))
                self.train_folds.append(np.load(os.path.join(self.model_path, 'train_fold{}.npy'.format(fold_idx))))
                self.validation_folds.append(np.load(os.path.join(self.model_path, 'validation_fold{}.npy'.format(fold_idx))))
            else:
                warnings.warn('Missing model for fold {}'.format(fold_idx))
                self.models.append(None)
                self.train_folds.append(None)
                self.validation_folds.append(None)
            self.folds[fold_idx] = try_load(os.path.join(self.model_path, 'test_fold{}.npy'.format(fold_idx)),
                                             np.load,
                                             fail=lambda _: self.folds[fold_idx])
        mu_path = os.path.join(self.model_path, 'mu.npy')
        if os.path.exists(mu_path):
            self.mu = np.load(mu_path)
        else:
            warnings.warn('Missing out-of-sample mu values')


'''Use a simple neural regression model'''
class NeuralModel(nn.Module):
    def __init__(self, nfeatures, ndrugs, ndoses, layers=None, dropout=True, batchnorm=True, **kwargs):
        super(NeuralModel, self).__init__()
        self.nfeatures = nfeatures
        self.ndrugs = ndrugs
        self.ndoses = ndoses
        self.nout = ndrugs*ndoses

        # Setup the NN layers
        all_layers = []
        prev_out = nfeatures
        if layers is not None:
            for layer_size in layers:
                if dropout:
                    all_layers.append(nn.Dropout())
                if batchnorm:
                    all_layers.append(nn.BatchNorm1d(prev_out))
                all_layers.append(nn.Linear(prev_out, layer_size))
                all_layers.append(nn.ReLU())
                prev_out = layer_size
        # if dropout:
        #     all_layers.append(nn.Dropout())
        # if batchnorm:
        #     all_layers.append(nn.BatchNorm1d(prev_out))
        all_layers.append(nn.Linear(prev_out, self.nout))
        self.fc_in = nn.Sequential(*all_layers)
        self.softplus = nn.Softplus()
    
    def forward(self, X):
        fwd = self.fc_in(X).reshape(-1, self.ndrugs, self.ndoses)

        # Enforce monotonicity
        mu = torch.cat([fwd[:,:,0:1], fwd[:,:,0:1] + self.softplus(fwd[:,:,1:]).cumsum(dim=2)], dim=2)

        # Do we want gradients or just predictions?
        if self.training:
            # Reparameterization trick for beta with diagonal covariance
            # Z = np.random.normal(0,1,size=mu.shape)
            # noise = autograd.Variable(torch.FloatTensor(Z), requires_grad=False)
            noise = 0 # TEMP

            # Get the MVN draw as mu + epsilon
            beta = mu + noise
        else:
            beta = mu

        # Logistic transform on the log-odds prior sample
        tau = 1 / (1. + (-beta).exp())
        # tau = 1 / (1 + nn.Softplus()(beta))

        return tau

def create_predictive_model(model_save_path, genomic_features, drug_responses, drug_details,
                          feature_types=['MUT', 'CNV', 'EXP', 'TISSUE'], no_fix=False,
                          **kwargs):
    print('Loading genomic features')
    X = load_dataset(genomic_features, index_col=0)

    # Remove any features not specified (this is useful for ablation studies)
    for ftype in ['MUT', 'CNV', 'EXP', 'TISSUE']:
        if ftype not in feature_types:
            select = [c for c in X.columns if not c.startswith(ftype)]
            print('Removing {} {} features'.format(X.shape[1] - len(select), ftype))
            X = X[select]

    feature_names = X.columns

    print('Loading response data')
    df = load_dataset(drug_responses) # usually data/raw_step3.csv

    # Get the observations
    treatment_cols = ['raw_max'] + ['raw{}'.format(i) for i in range(2,10)]
    Y_raw = df[treatment_cols].values
    a_raw = df['Pos_MLE_Shape'].values
    b_raw = df['Pos_MLE_Scale'].values
    c_raw = df['Neg_MAP_Estimate'].values
    
    # Handle some idiosyncracies of the GDSC dataset
    if no_fix:
        import warnings
        warnings.warn('Fix dosages is not enabled. GDSC data requires fixing; this should only be specified on another dataset.')
    else:
        select = np.any(np.isnan(Y_raw), axis=1)
        Y_raw[select,0::2] = Y_raw[select,:5]
        Y_raw[select,1::2] = np.nan

    # Transform the dataset into a multi-task regression one
    print('Building multi-task response')
    raw_index = np.full((Y_raw.shape[0], 2), -1, dtype=int)
    cell_ids = {c: i for i,c in enumerate(X.index)}
    drug_ids = {d: i for i,d in enumerate(df['DRUG_ID'].unique())}
    # cosmic_ids = {row['CELL_LINE_NAME']: row['COSMIC_ID'] for idx, row in df[['CELL_LINE_NAME', 'COSMIC_ID']].drop_duplicates().iterrows()}
    Y = np.full((len(cell_ids), len(drug_ids), Y_raw.shape[1]), np.nan)
    A = np.full(Y.shape[:2], np.nan)
    B = np.full(Y.shape[:2], np.nan)
    C = np.full(Y.shape[:2], np.nan)
    missing = set()
    missing_cosmic = set()
    for idx, row in df.iterrows():
        cell_id = row['CELL_LINE_NAME']
        drug_id = row['DRUG_ID']
        # Skip cell lines that have no features
        if cell_id not in cell_ids:
            missing.add(cell_id)
            continue
        i, j = cell_ids[cell_id], drug_ids[drug_id]
        Y[i,j] = Y_raw[idx]
        A[i,j] = a_raw[idx]
        B[i,j] = b_raw[idx]
        C[i,j] = c_raw[idx]
        raw_index[idx] = i, j

    # print('Y shape: {} Missing responses: {}'.format(Y.shape, np.isnan(Y).sum()))
    # print('Missing any response: {}'.format(np.all(np.all(np.isnan(Y), axis=-1), axis=-1).sum()))
    # print('Mismatched cell lines:')
    # print(sorted(missing))

    # Remove the cell lines missing any responses
    all_missing = np.all(np.all(np.isnan(Y), axis=-1), axis=-1)
    print('Removing cell lines with no response data:')
    print(sorted(X.index[all_missing]))
    X = X.iloc[~all_missing]
    Y = Y[~all_missing]
    A = A[~all_missing]
    B = B[~all_missing]
    C = C[~all_missing]
    raw_index[:,0] -= (raw_index[:,0:1] >= (np.arange(len(all_missing))[all_missing])[None]).sum(axis=1)

    print('Loading drug names')
    drug_idx = [None for _ in range(len(drug_ids))]
    for d,i in drug_ids.items():
        drug_idx[i] = d
    drug_names = pd.read_csv(drug_details, index_col=0, header=0).loc[drug_idx]['Drug Name'].values
    
    print('Building optimizer')
    ebo = EmpiricalBayesOptimizer()
    ebo.setup(model_save_path, X.index, drug_names, list(drug_ids.keys()), feature_names,
                X.values, Y, A, B, C, raw_index, **kwargs)

    return ebo


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    import argparse

    parser = argparse.ArgumentParser(description='Deep empirical Bayes dose-response model fitting.')

    # Experiment settings
    parser.add_argument('name', default='gdsc', help='The project name. Will be prepended to plots and saved files.')
    parser.add_argument('--drug_responses', default='data/raw_step3.csv', help='The dataset file with all of the experiments.')
    parser.add_argument('--genomic_features', default='data/gdsc_all_features.csv', help='The file with the cell line features.')
    parser.add_argument('--drug_details', default='data/gdsc_drug_details.csv', help='The data file with all of the drug information (names, targets, etc).')
    parser.add_argument('--plot_path', default='plots', help='The path where plots will be saved.')
    parser.add_argument('--save_path', default='data', help='The path where data and models will be saved.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--torch_threads', type=int, default=1, help='The number of threads that pytorch can use in a fold.')
    parser.add_argument('--no_fix', action='store_true', default=False, help='Do not correct the dosages.')
    parser.add_argument('--nepochs', type=int, default=50, help='The number of training epochs per fold.')
    parser.add_argument('--nfolds', type=int, default=10, help='The number of cross validation folds.')
    parser.add_argument('--batch_size', type=int, default=10, help='The mini-batch size.')
    parser.add_argument('--lr', type=float, default=3e-4, help='The SGD learning rate.')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'RMSprop'], default='RMSprop', help='The type of SGD method.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='The weight decay for SGD.')
    parser.add_argument('--step_decay', type=float, default=0.998, help='The exponential decay for the learning rate per epoch.')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum, if applicable.')
    parser.add_argument('--layers', type=int, nargs='*', default=[1000,200,200], help='The hidden layer dimensions of the NN.')
    parser.add_argument('--feature_types', choices=['MUT', 'CNV', 'EXP', 'TISSUE'], nargs='*', default=['MUT', 'CNV', 'EXP', 'TISSUE'], help='The type of genomic features to use. By default we use the full feature set.')
    parser.add_argument('--fold', type=int, help='If specified, trains only on a specific cross validation fold. This is useful for parallel/distributed training.')
    parser.add_argument('--checkpoint', action='store_true', help='If specified, saves progress after every epoch of training.')
    parser.add_argument('--verbose', action='store_true', help='If specified, prints progress to terminal.')
    
    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Seed the random number generators so we get reproducible results
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    print('Running step 4 with args:')
    print(args)
    print('Using feature set {}'.format(args.feature_types))
    print('Working on project: {}'.format(args.name))

    # Create the model directory
    model_save_path = os.path.join(args.save_path, args.name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Load the predictor
    ebo = create_predictive_model(model_save_path, **dargs)

    print('Fitting model')
    ebo.fit_mu(**dargs)

    # print('Loading model from file')
    # ebo.load()
    
    print('Plotting out of sample examples')
    for i in range(5):
        for j in range(5):
            ebo._predict_mu_insample(0,i,j)












