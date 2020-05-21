import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


class LinearDrugResponsePrior(nn.Module):
    def __init__(self, responses, genomic_features=None, ndoses=9):
        super(LinearDrugResponsePrior, self).__init__()
        self.ndoses = ndoses
        self.noutputs = ndoses

        if genomic_features is not None:
            self.nfeatures = len(genomic_features.iloc[:,0])

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
            
            # Find all the missing cell lines
            print('\tFinding missing cell lines')
            self.missing_cells = list(set(responses['CELL_LINE_NAME'].unique()) - set(genomic_features.columns))
            self.is_missing = np.array([1 if row['CELL_LINE_NAME'] in self.missing_cells else 0 for i,row in responses.iterrows()], dtype=int)
            
            nmissing = len(self.missing_cells)
            print('\tFound {} missing cell lines.'.format(nmissing))

            # Map from the example index to either the features or the embedding
            print('\tMapping from cell lines to features and embeddings')
            self.cell_map = np.array([self.missing_cells.index(c) if m else self.cell_cols.index(c) for m,c in zip(self.is_missing, responses['CELL_LINE_NAME'])])
            
            # Build the mutation feature component
            print('\tBuilding torch model')
            self.mut_features = autograd.Variable(torch.FloatTensor(np.random.normal(0,0.01,size=(self.nfeatures, self.noutputs))))
            # self.mut_features.weight.data.copy_(torch.from_numpy(np.random.normal(0,0.01,size=(self.noutputs, self.nfeatures))))

            # Create embeddings for all the cell lines without mutation data
            self.missing_embeddings = nn.Embedding(nmissing, self.noutputs)
            # self.missing_embeddings.weight.data.copy_(torch.from_numpy(np.random.normal(0,0.01,size=(nmissing, self.noutputs))))
        else:
            self.nfeatures = 0

        
        # Create embeddings for all the drugs
        self.drug_ids = {d: i for i,d in enumerate(responses['DRUG_ID'].unique())}
        ndrugs = len(self.drug_ids)
        self.drug_embeddings = nn.Embedding(ndrugs, self.noutputs)
        self.drug_map = autograd.Variable(torch.LongTensor([self.drug_ids[d] for d in responses['DRUG_ID']]), requires_grad=False)

        # Softplus activation to make the means monotonic
        self.mean_sp = nn.Softplus()
        
    def forward(self, idx, tidx):
        cell_embeds = self.cell_lookup(idx) if self.nfeatures > 0 else None
        # print(cell_embeds.shape, cell_embeds.data.numpy().min(), cell_embeds.data.numpy().max())
        drug_embeds = self.drug_lookup(tidx)
        return self.embeds_to_predictions(cell_embeds, drug_embeds)

    def embeds_to_predictions(self, cell_embeds, drug_embeds):
        mu_cell = 0 if cell_embeds is None else self.fwd_to_mu(cell_embeds)
        mu_drug = self.fwd_to_mu(drug_embeds)
        return mu_cell + mu_drug

    def fwd_to_mu(self, fwd):
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
                t = torch.matmul(cf, self.mut_features)
            results.append(t)
        return torch.cat(results).view(-1, self.noutputs)

    def drug_lookup(self, indices):
        return self.drug_embeddings(self.drug_map[indices])

    def predict(self, cell_features, drug_ids):
        self.eval()
        torch_cell_features = autograd.Variable(torch.FloatTensor((cell_features - self.genomic_feat_mean[np.newaxis, :]) / self.genomic_feat_std[np.newaxis, :]), requires_grad=False)
        torch_drug_ids = autograd.Variable(torch.LongTensor(drug_ids), requires_grad=False)
        cell_embeds = self.mut_features(torch_cell_features)
        drug_embeds = self.drug_embeddings(torch_drug_ids)
        mu = self.embeds_to_predictions(cell_embeds, drug_embeds)
        mu = mu.data.numpy()
        return mu





