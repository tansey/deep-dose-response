'''STEP 4 fit a deep empirical Bayes prior model via SGD.

Builds an empirical Bayes model to predict the prior over the dose-response
mean-effect curve.

We use a neural network to model, trained with stochastic gradient descent.
The features are the mutation, copy number, and gene expression information as
well as the ID of each drug. We use an embedding model for drugs and a separate
neural embedding model for any missing cell lines. The two embeddings are then
passed through a neural network to output the mean and covariance for the MVN
prior on the dose-response.

For numerical purposes, we approximate the double integral by a finite grid
over lambda and Monte Carlo sampling for beta.
'''
from __future__ import print_function
import sys
import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from slice_samplers import posterior_ess_Sigma
from utils import load_dataset, batches, create_folds, \
                  pretty_str, pav, ilogit, logsumexp

class EmpiricalBayesOptimizer:
    def __init__(self, Y=None,
                       a=None, b=None, c=None, # Hyperparameters estimated offline
                       lam_gridsize=100, # Number of discrete points to approximate the NLL integral with
                       restore_path=None,
                       lam_path=None,
                       **kwargs):
        if Y is not None:
            self.Y_shape = Y.shape
            self.nsamples = Y.shape[0]
            self.ndoses = Y.shape[1]

        self.a = a
        self.b = b
        self.c = c
        self.Y = Y
        self.lam_gridsize = lam_gridsize
        
        if restore_path is None:
            # cache which dosages are missing
            self.obs_mask = (~np.isnan(Y)).astype(int)

            if lam_path is None or not os.path.exists(os.path.join(lam_path, 'lam_grid.npy')):
                # We approximate the integral over lambda with a finite grid of lam_gridsize points
                print('Caching lambda integral approximation')
                from scipy.stats import gamma
                self.lam_grid = []
                self.lam_weights = []
                for i, (a_p, b_p) in enumerate(zip(a,b)):
                    grid = np.linspace(gamma.ppf(1e-3, a_p, scale=b_p), gamma.ppf(1-1e-3, a_p, scale=b_p), lam_gridsize)[np.newaxis,:]
                    weights = gamma.pdf(grid, a_p, scale=b_p)
                    weights /= weights.sum()
                    weights = np.log(weights.clip(1e-6,1-1e-6))
                    self.lam_grid.append(grid)
                    self.lam_weights.append(weights)
                self.lam_grid = np.array(self.lam_grid)
                self.lam_weights = np.array(self.lam_weights)
                if lam_path is not None:
                    print('Saving cached lambda integral approximations')
                    np.save(os.path.join(lam_path, 'lam_grid.npy'), self.lam_grid)
                    np.save(os.path.join(lam_path, 'lam_weights.npy'), self.lam_weights)
            else:
                print('Loading lambda integral approximations')
                self.lam_grid = np.load(os.path.join(lam_path, 'lam_grid.npy'))
                self.lam_weights = np.load(os.path.join(lam_path, 'lam_weights.npy'))
                assert self.lam_grid.shape[0] == self.Y.shape[0]

            print('Replacing missing dosages to prevent NaN propagation')
            for i, (a_p, b_p) in enumerate(zip(a,b)):
                from scipy.stats import gamma
                self.Y[i,np.isnan(self.Y[i])] = gamma.ppf(0.5, a_p, scale=b_p) + c[i]
        else:
            self.load(restore_path)

    def train(self, model_fn,
                    bandwidth=2., kernel_scale=0.35, variance=0.02,
                    mvn_train_samples=5, mvn_validate_samples=105,
                    validation_samples=1000,
                    validation_burn=1000,
                    validation_mcmc_samples=1000,
                    validation_thin=1,
                    lr=3e-4, num_epochs=10, batch_size=100,
                    val_pct=0.1, nfolds=5, folds=None,
                    learning_rate_decay=0.9, weight_decay=0.,
                    clip=None, group_lasso_penalty=0.,
                    save_dir='tmp/',
                    checkpoint=False,
                    target_fold=None):
        print('\tFitting model using {} folds and training for {} epochs each'.format(nfolds, num_epochs))
        torch_Y = autograd.Variable(torch.FloatTensor(self.Y), requires_grad=False)
        torch_lam_grid = autograd.Variable(torch.FloatTensor(self.lam_grid), requires_grad=False)
        torch_lam_weights = autograd.Variable(torch.FloatTensor(self.lam_weights), requires_grad=False)
        torch_c = autograd.Variable(torch.FloatTensor(self.c[:,np.newaxis,np.newaxis]), requires_grad=False)
        torch_obs = autograd.Variable(torch.FloatTensor(self.obs_mask), requires_grad=False)
        torch_dose_idxs = [autograd.Variable(torch.LongTensor(
                                np.arange(d+(d**2 - d)//2, (d+1)+((d+1)**2 - (d+1))//2)), requires_grad=False)
                                for d in range(self.ndoses)]

        # Use a fixed kernel
        Sigma = np.array([kernel_scale*(np.exp(-0.5*(i - np.arange(self.ndoses))**2 / bandwidth**2)) for i in np.arange(self.ndoses)]) + variance*np.eye(self.ndoses) # squared exponential kernel
        L = np.linalg.cholesky(Sigma)[np.newaxis,np.newaxis,:,:]

        # Use a fixed set of noise draws for validation
        Z = np.random.normal(size=(self.Y_shape[0], mvn_validate_samples, self.ndoses, 1))
        validate_noise = autograd.Variable(torch.FloatTensor(np.matmul(L, Z)[:,:,:,0]), requires_grad=False)

        self.folds = folds if folds is not None else create_folds(self.Y_shape[0], nfolds)
        nfolds = len(self.folds)
        self.fold_validation_indices = []
        self.prior_mu = np.full(self.Y_shape, np.nan, dtype=float)
        self.prior_Sigma = np.zeros((nfolds, self.ndoses, self.ndoses))
        self.train_losses, self.val_losses = np.zeros((nfolds,num_epochs)), np.zeros((nfolds,num_epochs))
        self.epochs_per_fold = np.zeros(nfolds, dtype=int)
        self.models = [None for _ in range(nfolds)]
        for fold_idx, test_indices in enumerate(self.folds):
            # Create train/validate splits
            mask = np.ones(self.Y_shape[0], dtype=bool)
            mask[test_indices] = False
            indices = np.arange(self.Y_shape[0], dtype=int)[mask]
            np.random.shuffle(indices)
            train_cutoff = int(np.round(len(indices)*(1-val_pct)))
            train_indices = indices[:train_cutoff]
            validate_indices = indices[train_cutoff:]
            torch_test_indices = autograd.Variable(torch.LongTensor(test_indices), requires_grad=False)
            self.fold_validation_indices.append(validate_indices)

            # If we are only training one specific fold, skip all the rest
            if target_fold is not None and target_fold != fold_idx:
                continue

            if checkpoint:
                self.load_checkpoint(save_dir, fold_idx)

            if self.models[fold_idx] is None:
                self.models[fold_idx] = model_fn()

            model = self.models[fold_idx]

            # Setup the optimizers
            # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
            for epoch in range(self.epochs_per_fold[fold_idx], num_epochs):
                print('\t\tFold {} Epoch {}'.format(fold_idx+1,epoch+1))
                train_loss = torch.Tensor([0])
                for batch_idx, batch in enumerate(batches(train_indices, batch_size)):
                    if batch_idx % 100 == 0:
                        print('\t\t\tBatch {}'.format(batch_idx))
                        sys.stdout.flush()

                    tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)
                    Z = np.random.normal(size=(len(batch), mvn_train_samples, self.ndoses, 1))
                    noise = autograd.Variable(torch.FloatTensor(np.matmul(L, Z)[:,:,:,0]), requires_grad=False)

                    # Set the model to training mode
                    model.train()

                    # Reset the gradient
                    model.zero_grad()

                    # Run the model and get the prior predictions
                    mu = model(batch, tidx)

                    #### Calculate the loss as the negative log-likelihood of the data ####
                    # Get the MVN draw as mu + L.T.dot(Z)
                    beta = mu.view(-1,1,self.ndoses) + noise

                    # Logistic transform on the log-odds prior sample
                    tau = 1 / (1. + (-beta).exp())

                    # Poisson noise model for observations
                    rates = tau[:,:,:,None] * torch_lam_grid[tidx,None,:,:] + torch_c[tidx,None,:,:]
                    likelihoods = torch.distributions.Poisson(rates)

                    # Get log probabilities of the data and filter out the missing observations
                    loss = -(logsumexp(likelihoods.log_prob(torch_Y[tidx][:,None,:,None]) + torch_lam_weights[tidx][:,None,:,:], dim=-1).mean(dim=1) * torch_obs[tidx]).mean()

                    if group_lasso_penalty > 0:
                        loss += group_lasso_penalty * torch.norm(model.cell_line_features.weight, 2, 0).mean()

                    # Update the model
                    loss.backward()
                    if clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                        for p in model.parameters():
                            p.data.add_(-lr, p.grad.data)
                    else:
                        optimizer.step()

                    train_loss += loss.data

                validate_loss = torch.Tensor([0])
                for batch_idx, batch in enumerate(batches(validate_indices, batch_size, shuffle=False)):
                    if batch_idx % 100 == 0:
                        print('\t\t\tValidation Batch {}'.format(batch_idx))
                        sys.stdout.flush()
                    
                    tidx = autograd.Variable(torch.LongTensor(batch), requires_grad=False)
                    noise = validate_noise[tidx]

                    # Set the model to training mode
                    model.eval()

                    # Reset the gradient
                    model.zero_grad()

                    # Run the model and get the prior predictions
                    mu = model(batch, tidx)

                    #### Calculate the loss as the negative log-likelihood of the data ####
                    # Get the MVN draw as mu + L.T.dot(Z)
                    beta = mu.view(-1,1,self.ndoses) + noise

                    # Logistic transform on the log-odds prior sample
                    tau = 1 / (1. + (-beta).exp())

                    # Poisson noise model for observations
                    rates = tau[:,:,:,None] * torch_lam_grid[tidx,None,:,:] + torch_c[tidx,None,:,:]
                    likelihoods = torch.distributions.Poisson(rates)

                    # Get log probabilities of the data and filter out the missing observations
                    loss = -(logsumexp(likelihoods.log_prob(torch_Y[tidx][:,None,:,None]) + torch_lam_weights[tidx][:,None,:,:], dim=-1).mean(dim=1) * torch_obs[tidx]).sum()

                    validate_loss += loss.data

                self.train_losses[fold_idx, epoch] = train_loss.numpy() / float(len(train_indices))
                self.val_losses[fold_idx, epoch] = validate_loss.numpy() / float(len(validate_indices))

                # Adjust the learning rate down if the validation performance is bad
                scheduler.step(self.val_losses[fold_idx, epoch])

                # Check if we currently have the best held-out log-likelihood
                if epoch == 0 or np.argmin(self.val_losses[fold_idx, :epoch+1]) == epoch:
                    print('\t\t\tNew best score: {}'.format(self.val_losses[fold_idx,epoch]))
                    print('\t\t\tSaving test set results.')
                    # If so, use the current model on the test set
                    mu = model(test_indices, torch_test_indices)
                    self.prior_mu[test_indices] = mu.data.numpy()
                    self.save_fold(save_dir, fold_idx)
                
                cur_mu = self.prior_mu[test_indices]
                print('First 10 data points: {}'.format(test_indices[:10]))
                print('First 10 prior means:')
                print(pretty_str(ilogit(cur_mu[:10])))
                print('Prior mean ranges:')
                for dose in range(self.ndoses):
                    print('{}: {} [{}, {}]'.format(dose,
                                                   ilogit(cur_mu[:,dose].mean()),
                                                   np.percentile(ilogit(cur_mu[:,dose]), 5),
                                                   np.percentile(ilogit(cur_mu[:,dose]), 95)))
                print('Best model score: {} (epoch {})'.format(np.min(self.val_losses[fold_idx,:epoch+1]), np.argmin(self.val_losses[fold_idx, :epoch+1])+1))
                print('Current score: {}'.format(self.val_losses[fold_idx, epoch]))
                print('')

                self.epochs_per_fold[fold_idx] += 1
                
                # Update the save point if needed
                if checkpoint:
                    self.save_checkpoint(save_dir, fold_idx, model)
                    sys.stdout.flush()
                
            
            # Reload the best model
            tmp = model.cell_features
            self.load_fold(save_dir, fold_idx)
            self.models[fold_idx].cell_features = tmp

            print('Finished fold {}. Estimating covariance matrix using elliptical slice sampler with max {} samples.'.format(fold_idx+1, validation_samples))
            validate_subset = np.random.choice(validate_indices, validation_samples, replace=False) if len(validate_indices) > validation_samples else validate_indices
            tidx = autograd.Variable(torch.LongTensor(validate_subset), requires_grad=False)
                        
            # Set the model to training mode
            self.models[fold_idx].eval()

            # Reset the gradient
            self.models[fold_idx].zero_grad()

            # Run the model and get the prior predictions
            mu_validate = self.models[fold_idx](validate_subset, tidx).data.numpy()
            
            # Run the slice sampler to get the covariance and data log-likelihoods
            Y_validate = self.Y[validate_subset].astype(int)
            Y_validate[self.obs_mask[validate_subset] == 0] = -1
            (Beta_samples,
                Sigma_samples,
                Loglikelihood_samples) = posterior_ess_Sigma(Y_validate,
                                                             mu_validate,
                                                             self.a[validate_subset],
                                                             self.b[validate_subset],
                                                             self.c[validate_subset],
                                                             Sigma=Sigma,
                                                             nburn=validation_burn,
                                                             nsamples=validation_mcmc_samples,
                                                             nthin=validation_thin,
                                                             print_freq=1)

            # Save the result
            self.prior_Sigma[fold_idx] = Sigma_samples.mean(axis=0)
            print('Last sample:')
            print(pretty_str(Sigma_samples[-1]))
            print('Mean:')
            print(pretty_str(self.prior_Sigma[fold_idx]))

            if checkpoint:
                self.clean_checkpoint(save_dir, fold_idx)

        print('Finished training.')
        
        return {'train_losses': self.train_losses,
                'validation_losses': self.val_losses,
                'mu': self.prior_mu,
                'Sigma': self.prior_Sigma,
                'models': self.models}

    def predict(self, cell_features, drug_ids):
        mu = []
        for model in self.models:
            mu_i = model.predict(cell_features, drug_ids)
            mu.append(mu_i)
        return np.mean(mu, axis=0), self.prior_Sigma.mean(axis=0)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for fold_idx in range(len(self.folds)):
            self.save_fold(path, fold_idx)
        self.save_indices(path)

    def save_indices(self, path):
        import csv
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path,'folds.csv'), 'w') as f:
            writer = csv.writer(f)
            for fold in self.folds:
                writer.writerow(fold)
        with open(os.path.join(path,'validation_indices.csv'), 'w') as f:
            writer = csv.writer(f)
            for indices in self.fold_validation_indices:
                writer.writerow(indices)

    def save_fold(self, path, fold_idx):
        if not os.path.exists(path):
            os.makedirs(path)
        fold = self.folds[fold_idx]
        model = self.models[fold_idx]
        tmp = model.cell_features
        
        # Save the model but don't re-save the data (space saver)
        model.cell_features = None
        torch.save(model, os.path.join(path, 'model_fold{}.pt'.format(fold_idx)))
        model.cell_features = tmp

        # Save the model testing outputs for this fold
        np.save(os.path.join(path, 'prior_mu_fold{}'.format(fold_idx)), self.prior_mu[fold])
        np.save(os.path.join(path, 'prior_sigma_fold{}'.format(fold_idx)), self.prior_Sigma[fold_idx])
        np.save(os.path.join(path, 'train_losses_fold{}'.format(fold_idx)), self.train_losses[fold_idx])
        np.save(os.path.join(path, 'val_losses_fold{}'.format(fold_idx)), self.val_losses[fold_idx])

    def save_checkpoint(self, path, fold_idx, model):
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the performance scores
        np.save(os.path.join(path, 'val_losses_fold{}'.format(fold_idx)), self.val_losses[fold_idx])
        np.save(os.path.join(path, 'train_losses_fold{}'.format(fold_idx)), self.train_losses[fold_idx])
        np.save(os.path.join(path, 'epochs_fold{}'.format(fold_idx)), self.epochs_per_fold[fold_idx])

        # Save the model
        torch.save(self.models[fold_idx], os.path.join(path, 'model_checkpoint_fold{}.pt'.format(fold_idx)))

        # Save the outputs
        np.save(os.path.join(path, 'prior_mu_checkpoint_fold{}'.format(fold_idx)), self.prior_mu[self.folds[fold_idx]])

    def load(self, path):
        self.load_indices(path)
        self.models = [None for _ in self.folds]
        self.train_losses = [None for _ in self.folds]
        self.val_losses = [None for _ in self.folds]
        self.prior_mu = None
        self.prior_Sigma = None
        for fold_idx in range(len(self.folds)):
            self.load_fold(path, fold_idx)

    def load_indices(self, path):
        import csv
        self.folds = []
        self.fold_validation_indices = []
        with open(os.path.join(path,'folds.csv'), 'r') as f:
            reader = csv.reader(f)
            self.folds = [np.array([int(idx) for idx in line], dtype=int) for line in reader]
        with open(os.path.join(path,'validation_indices.csv'), 'r') as f:
            reader = csv.reader(f)
            self.fold_validation_indices = [np.array([int(idx) for idx in line], dtype=int) for line in reader]

    def load_fold(self, path, fold_idx):
        fold = self.folds[fold_idx]
        self.models[fold_idx] = torch.load(os.path.join(path, 'model_fold{}.pt'.format(fold_idx)))
        mu = np.load(os.path.join(path,'prior_mu_fold{}.npy'.format(fold_idx)))
        Sigma = np.load(os.path.join(path,'prior_sigma_fold{}.npy'.format(fold_idx)))

        # Initialize if not already done
        if self.prior_mu is None:
            self.prior_mu = np.zeros((max([max(idxs) for idxs in self.folds])+1, mu.shape[1]))
        if self.prior_Sigma is None:
            self.prior_Sigma = np.zeros((len(self.folds), mu.shape[1], mu.shape[1]))

        self.prior_mu[fold] = mu
        self.prior_Sigma[fold_idx] = Sigma
        self.val_losses[fold_idx] = np.load(os.path.join(path, 'val_losses_fold{}.npy'.format(fold_idx)))
        self.train_losses[fold_idx] = np.load(os.path.join(path, 'train_losses_fold{}.npy'.format(fold_idx)))

    def load_checkpoint(self, path, fold_idx):
        # If there's no checkpoint, just return
        if not os.path.exists(os.path.join(path, 'model_checkpoint_fold{}.pt'.format(fold_idx))):
            return
        # Load the performance scores
        self.val_losses[fold_idx] = np.load(os.path.join(path, 'val_losses_fold{}.npy'.format(fold_idx)))
        self.train_losses[fold_idx] = np.load(os.path.join(path, 'train_losses_fold{}.npy'.format(fold_idx)))
        self.epochs_per_fold[fold_idx] = np.load(os.path.join(path, 'epochs_fold{}.npy'.format(fold_idx)))

        # Load the model
        self.models[fold_idx] = torch.load(os.path.join(path, 'model_checkpoint_fold{}.pt'.format(fold_idx)))

        # Load the predictions
        self.prior_mu[self.folds[fold_idx]] = np.load(os.path.join(path, 'prior_mu_checkpoint_fold{}.npy'.format(fold_idx)))

    def clean_checkpoint(self, path, fold_idx):
        if os.path.exists(os.path.join(path, 'model_checkpoint_fold{}.pt'.format(fold_idx))):
            os.remove(os.path.join(path, 'model_checkpoint_fold{}.pt'.format(fold_idx)))
        if os.path.exists(os.path.join(path, 'prior_mu_checkpoint_fold{}.npy'.format(fold_idx))):
            os.remove(os.path.join(path, 'prior_mu_checkpoint_fold{}.npy'.format(fold_idx)))

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.reslayer = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bn(self.reslayer(x))
        out += residual
        return self.relu(out)

class DrugResponsePrior(nn.Module):
    def __init__(self, genomic_features, responses, ndoses=9, cell_embedding_size=1000, drug_embedding_size=100, resnet=False):
        super(DrugResponsePrior, self).__init__()
        self.nfeatures = genomic_features.shape[0]
        self.ndoses = ndoses
        self.noutputs = ndoses
        self.cell_embedding_size = cell_embedding_size
        self.drug_embedding_size = drug_embedding_size

        if self.nfeatures > 0:        
            # Create the matrix of features
            self.genomic_feat_mean = genomic_features.values.mean(axis=1)
            self.genomic_feat_std = genomic_features.values.std(axis=1)
            if self.genomic_feat_std.min() == 0:
                print('WARNING: features with zero variance detected: {}'.format(genomic_features.index[genomic_feat_std == 0]))
                print('These features will add no information to your model and should be removed.')
                self.genomic_feat_std[self.genomic_feat_std == 0] = 1 # Handle constant features
            self.cell_cols = list(genomic_features.columns)
            self.cell_features = autograd.Variable(torch.FloatTensor((genomic_features.values.T - self.genomic_feat_mean[np.newaxis, :]) / self.genomic_feat_std[np.newaxis, :]), requires_grad=False)
            print('\tHave {} features for {} cells lines measured at (max) {} doses'.format(self.nfeatures, len(self.cell_cols), ndoses))
        
            # Build the mutation feature component
            print('\tBuilding torch model')
            self.cell_line_features = nn.Sequential(nn.Linear(self.nfeatures, cell_embedding_size), nn.ReLU(), nn.Dropout())

            cell_lines = set(genomic_features.columns)
        else:
            cell_lines = set()
            self.cell_features = None
            self.cell_line_features = None
            self.cell_cols = None
            self.genomic_feat_std = None
            self.genomic_feat_mean = None

        # Find all the missing cell lines
        print('\tFinding missing cell lines')
        self.missing_cells = list(set(responses['CELL_LINE_NAME'].unique()) - cell_lines)
        self.is_missing = np.array([1 if row['CELL_LINE_NAME'] in self.missing_cells else 0 for i,row in responses.iterrows()], dtype=int)
        
        nmissing = len(self.missing_cells)
        print('\tFound {} missing cell lines. Using an embedding of size cells={}, drugs={}'.format(nmissing, cell_embedding_size, drug_embedding_size))

        # Map from the example index to either the features or the embedding
        print('\tMapping from cell lines to features and embeddings')
        self.cell_map = np.array([self.missing_cells.index(c) if m else self.cell_cols.index(c) for m,c in zip(self.is_missing, responses['CELL_LINE_NAME'])])

        # Create embeddings for all the cell lines without mutation data
        self.missing_embeddings = nn.Embedding(nmissing, cell_embedding_size)
        # self.missing_embeddings.weight.data.copy_(torch.from_numpy(np.random.normal(0,0.01,size=(nmissing, embedding_size))))

        
        # Create embeddings for all the drugs
        self.drug_ids = {d: i for i,d in enumerate(responses['DRUG_ID'].unique())}
        ndrugs = len(self.drug_ids)
        self.drug_embeddings = nn.Embedding(ndrugs, drug_embedding_size)
        self.drug_map = autograd.Variable(torch.LongTensor([self.drug_ids[d] for d in responses['DRUG_ID']]), requires_grad=False)

        # Combine the cell and drug embeddings to produce a prior mean
        if resnet:
            self.feed_forward = nn.Sequential(
                                    nn.Linear(cell_embedding_size+drug_embedding_size, 200),
                                    nn.BatchNorm1d(200),
                                    nn.ReLU(),
                                    ResidualBlock(200),
                                    ResidualBlock(200),
                                    ResidualBlock(200),
                                    ResidualBlock(200),
                                    nn.Linear(200, self.noutputs))
        else:
            self.feed_forward = nn.Sequential(
                                    nn.Linear(cell_embedding_size+drug_embedding_size, 200),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(200, 200),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(200, self.noutputs))
        

        # Softplus activation to make the means monotonic
        self.mean_sp = nn.Softplus()
        
    def forward(self, idx, tidx):
        cell_embeds = self.cell_lookup(idx)
        drug_embeds = self.drug_lookup(tidx)
        return self.embeds_to_predictions(cell_embeds, drug_embeds)

    def embeds_to_predictions(self, cell_embeds, drug_embeds):
        cell_embeds = nn.functional.normalize(cell_embeds, p=2, dim=1)
        drug_embeds = nn.functional.normalize(drug_embeds, p=2, dim=1)
        fwd = self.feed_forward(torch.cat([cell_embeds, drug_embeds], 1)) # N x noutputs
        mu = torch.cat([fwd[:,0:1], fwd[:,0:1] + self.mean_sp(fwd[:,1:]).cumsum(dim=1)], dim=1) # Enforce monotonicity
        return mu

    # Use the features if they exist, otherwise look up the embeddings
    def cell_lookup(self, indices):
        results = []
        for i,idx in enumerate(indices):
            cm = self.cell_map[idx]
            if self.is_missing[idx]:
                t = self.missing_embeddings(autograd.Variable(torch.LongTensor([int(cm)])))[0]
            else:
                cf = self.cell_features[cm]
                t = self.cell_line_features(cf)
            results.append(t)
        return torch.cat(results).view(-1, self.cell_embedding_size)

    def drug_lookup(self, indices):
        return self.drug_embeddings(self.drug_map[indices])

    def get_cell_embeddings(self):
        results = []
        names = []
        for i,name in enumerate(self.cell_cols):
            results.append(self.cell_line_features(self.cell_features[i]).data.numpy())
            names.append(name)
        for i,name in enumerate(self.missing_cells):
            results.append(self.missing_embeddings.weight[i].data.numpy())
            names.append(name)
        return np.array(results), names

    def get_drug_embeddings(self):
        drug_names = [None]*len(self.drug_ids)
        for d,i in self.drug_ids.items():
            drug_names[i] = d
        return self.drug_embeddings.weight.data.numpy(), drug_names

    def predict(self, cell_features, drug_ids):
        self.eval()
        torch_cell_features = autograd.Variable(torch.FloatTensor((cell_features - self.genomic_feat_mean[np.newaxis, :]) / self.genomic_feat_std[np.newaxis, :]), requires_grad=False)
        torch_drug_ids = autograd.Variable(torch.LongTensor(np.array([self.drug_ids[d] for d in drug_ids])), requires_grad=False)
        cell_embeds = self.cell_line_features(torch_cell_features)
        drug_embeds = self.drug_embeddings(torch_drug_ids)
        mu = self.embeds_to_predictions(cell_embeds, drug_embeds)
        mu = mu.data.numpy()
        return mu
        

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    import sys
    import os
    import argparse

    '''
    Standard setup:
     python python/step4_fit_empirical_bayes_prior.py --checkpoint --name gdsc_mut_cnv_exp_embed30 --feature_types MUT CNV EXP --fold 0
    '''
    parser = argparse.ArgumentParser(description='Deep empirical Bayes dose-response model fitting.')

    # Experiment settings
    parser.add_argument('--name', default='gdsc', help='The project name. Will be prepended to plots and saved files.')
    parser.add_argument('--dataset', default='data/raw_step3.csv', help='The dataset file with all of the experiments.')
    parser.add_argument('--genomic_features', default='data/gdsc_all_features.csv', help='The file with the cell line features.')
    parser.add_argument('--drug_features', default='data/gdsc_mol2vec_features.csv', help='The file with the drug features.')
    parser.add_argument('--plot_path', default='plots', help='The path where plots will be saved.')
    parser.add_argument('--save_path', default='data', help='The path where data and models will be saved.')
    parser.add_argument('--nepochs', type=int, default=50, help='The number of training epochs per fold.')
    parser.add_argument('--nfolds', type=int, default=10, help='The number of cross validation folds.')
    parser.add_argument('--batch_size', type=int, default=100, help='The mini-batch size.')
    parser.add_argument('--lr', type=float, default=3e-4, help='The SGD learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='The weight decay for SGD.')
    parser.add_argument('--clip', type=float, help='If specified, use gradient clipping at the specified amount.')
    parser.add_argument('--mvn_train_samples', type=int, default=1, help='Sample size for training gradients.')
    parser.add_argument('--mvn_validate_samples', type=int, default=100, help='Sample size for validation gradients.')
    parser.add_argument('--validation_samples', type=int, default=1000, help='Maximum number of samples to use in the post-fitting uncertainty step.')
    parser.add_argument('--validation_burn', type=int, default=1000, help='Number of burn-in steps for validation MCMC sampler.')
    parser.add_argument('--validation_mcmc_samples', type=int, default=1000, help='Number of samples for validation MCMC sampler.')
    parser.add_argument('--validation_thin', type=int, default=1, help='Number of thinning steps for validation MCMC sampler.')
    parser.add_argument('--cell_embedding_size', type=int, default=1000, help='The number of embedding dimensions for cell lines.')
    parser.add_argument('--drug_embedding_size', type=int, default=100, help='The number of embedding dimensions for drugs.')
    parser.add_argument('--group_lasso', type=float, default=0, help='The group lasso penalty to apply to feature input weights.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--torch_threads', type=int, default=1, help='The number of threads that pytorch can use in a fold.')
    parser.add_argument('--no_fix', action='store_true', help='Correct the dosages if they are mixed 2x and 4x dilution.')
    parser.add_argument('--model_type', choices=['blackbox', 'linear', 'drug_only', 'drug_features'], default='blackbox', help='The type of prior model to use. By default we use the full blackbox model.')
    parser.add_argument('--feature_types', choices=['MUT', 'CNV', 'EXP', 'TISSUE'], nargs='*', default=['MUT', 'CNV', 'EXP', 'TISSUE'], help='The type of genomic features to use. By default we use the full feature set.')
    parser.add_argument('--fold', type=int, help='If specified, trains only on a specific cross validation fold. This is useful for parallel/distributed training.')
    parser.add_argument('--checkpoint', action='store_true', help='If specified, saves progress after every epoch of training.')
    parser.add_argument('--cell_line_folds', action='store_true', help='If specified, entire cell lines are held out in cross-validation.')
    parser.add_argument('--lam_path', default='data', help='The path to the lambda integral cache.')
    parser.add_argument('--resnet', action='store_true', help='If specified, uses a deep residual architecture instead of a simpler NN.')

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

    print('Loading genomic features')
    X = load_dataset(args.genomic_features, index_col=0).T # usually data/gdsc_raul_features_with_expression.csv

    for ftype in ['MUT', 'CNV', 'EXP', 'TISSUE']:
        if ftype not in args.feature_types:
            select = [c for c in X.index if not c.startswith(ftype)]
            print('Removing {} {} features'.format(X.shape[0] - len(select), ftype))
            X = X.loc[select]
    
    print('Loading response data')
    df = load_dataset(args.dataset) # usually data/raw_step3.csv

    # Get the observations
    treatment_cols = ['raw_max'] + ['raw{}'.format(i) for i in range(2,10)]
    Y = df[treatment_cols].values
    a = df['Pos_MLE_Shape'].values
    b = df['Pos_MLE_Scale'].values
    c = df['Neg_MAP_Estimate'].values
    
    # Handle some idiosyncracies of the GDSC dataset
    if args.no_fix:
        import warnings
        warnings.warn('Fix dosages is not enabled. GDSC data requires fixing; this should only be unspecified on another dataset.')
    else:
        select = np.any(np.isnan(Y), axis=1)
        Y[select,0::2] = Y[select,:5]
        Y[select,1::2] = np.nan

    print('Building {} prior'.format(args.model_type))
    if args.model_type == 'blackbox':
        model_fn = lambda: DrugResponsePrior(X, df, cell_embedding_size=args.cell_embedding_size,
                                                    drug_embedding_size=args.drug_embedding_size,
                                                    resnet=args.resnet)
    elif args.model_type == 'linear':
        from alternative_priors import LinearDrugResponsePrior
        model_fn = lambda: LinearDrugResponsePrior(df, genomic_features=X)
    elif args.model_type == 'drug_only':
        from alternative_priors import LinearDrugResponsePrior
        model_fn = lambda: LinearDrugResponsePrior(df)
    elif args.model_type == 'drug_features':
        from drug_features_prior import DrugResponsePrior as DrugFeaturePrior
        print('Loading drug features')
        Z = load_dataset(args.drug_features, index_col=0).T
        model_fn = lambda: DrugFeaturePrior(df,
                                        genomic_features=X,
                                        drug_features=Z,
                                        cell_embedding_size=args.cell_embedding_size,
                                        drug_embedding_size=args.drug_embedding_size)

    print('Building optimizer')
    ebo = EmpiricalBayesOptimizer(Y, a, b, c, lam_path=args.lam_path)

    if args.cell_line_folds:
        print('Creating cell line folds using only those with features')
        cell_lines_with_features = list(set(X.columns) & set(df['CELL_LINE_NAME'].unique()))
        cell_line_folds = create_folds(len(cell_lines_with_features), args.nfolds)
        cell_line_to_fold = {}
        for fold_idx, fold_cell_lines in enumerate(cell_line_folds):
            for c in fold_cell_lines:
                cell_line_to_fold[cell_lines_with_features[c]] = fold_idx
        folds = [[] for _ in range(args.nfolds)]
        for idx, c in enumerate(df['CELL_LINE_NAME']):
            if c in cell_line_to_fold:
                folds[cell_line_to_fold[c]].append(idx)
        for fold_idx, fold in enumerate(folds):
            print('Fold {}: {}'.format(fold_idx, len(fold)))
    else:
        folds = None

    print('Training model')
    results = ebo.train(model_fn, num_epochs=args.nepochs,
                                  nfolds=args.nfolds,
                                  folds=folds,
                                  batch_size=args.batch_size,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  clip=args.clip,
                                  mvn_train_samples=args.mvn_train_samples,
                                  mvn_validate_samples=args.mvn_validate_samples,
                                  validation_samples=args.validation_samples,
                                  validation_burn=args.validation_burn,
                                  validation_mcmc_samples=args.validation_mcmc_samples,
                                  validation_thin=args.validation_thin,
                                  group_lasso_penalty=args.group_lasso,
                                  save_dir=os.path.join(args.save_path, args.name),
                                  target_fold=args.fold,
                                  checkpoint=args.checkpoint)

    if args.fold is None:
        print('Saving complete model to file')
        ebo.save(os.path.join(args.save_path, args.name))
    else:
        if args.fold == 0:
            print('Saving indices')
            ebo.save_indices(os.path.join(args.save_path, args.name))
        ebo.save_fold(os.path.join(args.save_path, args.name), args.fold)


    print('Finished!')









