import numpy as np
import pandas as pd
from step4_fit_prior_fast import create_predictive_model, NeuralModel, EmpiricalBayesOptimizer
from utils import ilogit



if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Runs an amortized conditional independence test using targeted efficacy as the test statistic.')

    parser.add_argument('name', help='The project name. Will be prepended to plots and saved files.')
    parser.add_argument('--drug', default=-1, type=int, help='The feature to evaluate across all cell lines and drugs.')
    parser.add_argument('--toxicity', default=0.5, type=float, help='The cutoff for a drug being considered toxic.')
    parser.add_argument('--safety', default=0.8, type=float, help='The cutoff for a drug being considered safe.')
    parser.add_argument('--agg', action='store_true', help='Aggregate the results rather than evaluating a single feature.')
    parser.add_argument('--drug_responses', default='data/raw_step3.csv', help='The dataset file with all of the experiments.')
    parser.add_argument('--genomic_features', default='data/gdsc_all_features.csv', help='The file with the cell line features.')
    parser.add_argument('--drug_details', default='data/gdsc_drug_details.csv', help='The data file with all of the drug information (names, targets, etc).')
    parser.add_argument('--save_path', default='data/', help='The path where data and models will be saved.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--no_fix', action='store_true', default=False, help='Do not correct the dosages.')
    parser.add_argument('--ntrials', type=int, default=1000000, help='The number of random trials per feature when calculating p-values.')
    parser.add_argument('--fdr', type=float, default=0.2, help='The false discovery rate threshold to target.')
    parser.add_argument('--min_true', type=int, default=5, help='The minimum number of positive feature occurrences to test a feature.')
    
    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Seed the random number generators so we get reproducible results
    np.random.seed(args.seed)
    
    print('Running step 7 with args:')
    print(args)
    print('Working on project: {}'.format(args.name))

    # Create the model directory
    model_save_path = os.path.join(args.save_path, args.name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Process the data (easiest way to do this since it's all stored in ebo)
    ebo = create_predictive_model(model_save_path, **dargs)

    # Load the binarized features and factorization
    print('Loading binarized features')
    df_binarized = pd.read_csv(args.genomic_features.replace('.csv', '_binarized.csv'), header=0, index_col=0)
    W = np.load(args.genomic_features.replace('.csv', '_binarized_row_loading.npy'))
    V = np.load(args.genomic_features.replace('.csv', '_binarized_col_loading.npy'))

    # Get the features and probs
    X = df_binarized.values.astype(bool)
    X_probs = ilogit(W.dot(V.T))

    # Filter down to the cell lines we have features for
    indices = [i for i in range(X.shape[0]) if df_binarized.index[i] in ebo.cell_lines]
    X = X[indices]
    X_probs = X_probs[indices]

    # Filter down further to the cell lines we have responses with this drug
    indices = np.arange(ebo.Y.shape[0])[np.any(ebo.obs_mask[:,args.drug].astype(bool), axis=1)]
    X = X[indices]
    X_probs = X_probs[indices]

    # Load the posteriors for this drug
    Tau = np.load(os.path.join(model_save_path, 'posteriors/taus{}.npy'.format(args.drug)))

    # Cache the toxicity and safety probabilities for each cell line
    prob_safe = (Tau >= args.safety).mean(axis=0)
    prob_toxic = (Tau <= args.toxicity).mean(axis=0)

    print('Calculating targeted efficacy scores')
    targeted_efficacy_fn = lambda x: np.min([prob_toxic[x].mean(axis=0), prob_safe[~x].mean(axis=0)], axis=0).max(axis=0)
    p_values = np.full(X.shape[1], np.nan)

    # Iterate over features repeatedly, trying increasingly more trials to get finer-grained p-values
    ntrials = 100
    features = [(i,c) for i,c in enumerate([c for c in df_binarized.columns if not c.startswith('TISSUE')])]
    while len(features) > 0 and ntrials <= args.ntrials:
        print('\n\nEvaluating {} features with {} trials'.format(len(features), ntrials))

        # The list of features that need to be re-evaluated with more trials
        next_features = []

        for feat_idx, feature in features:
            # Skip rare features
            pos_true, neg_true = X[:,feat_idx].sum(), X.shape[0] - X[:,feat_idx].sum()
            if min(pos_true, neg_true) < args.min_true:
                continue

            # Baseline efficacy
            efficacy_true = max(targeted_efficacy_fn(X[:,feat_idx]), targeted_efficacy_fn(~X[:,feat_idx]))

            # Do everything in 100k batches for memory purposes
            batch_size = min(100000, ntrials)
            for batch in range(ntrials//batch_size + (ntrials%batch_size != 0)):
                # Sample a bunch of null replicates and evaluate each
                U = np.random.random(size=(X.shape[0], batch_size)) <= X_probs[:,feat_idx:feat_idx+1]
                npos = U.sum(axis=0)[None]
                nneg = X.shape[0] - npos
                npos, nneg = npos.clip(1,np.inf), nneg.clip(1,np.inf)
                efficacy_null = np.max(np.max([np.min([(prob_safe[:,:,None]*U[:,None]).sum(axis=0)/npos, (prob_toxic[:,:,None]*(1-U[:,None])).sum(axis=0)/nneg], axis=0),
                                    np.min([(prob_safe[:,:,None]*(1-U[:,None])).sum(axis=0)/nneg, (prob_toxic[:,:,None]*U[:,None]).sum(axis=0)/npos], axis=0)], axis=0), axis=0)

                p_value = (efficacy_null >= efficacy_true).sum()

            p_value /= ntrials

            print(feat_idx,
                    feature,
                    p_value,
                    '' if p_value > 0.1 else ('*' if p_value > 0.01 else ('**' if p_value > 0.001 else '***')),
                    '{} positive, {} negative'.format(pos_true, neg_true))

            p_values[feat_idx] = p_value

            # Keep high-scoring features to calculate more precise p-values
            if p_value < 10/ntrials:
                next_features.append((feat_idx, feature))

        # Increase the number of trials by an order of magnitude
        ntrials *= 10
        features = next_features

    # Save the results to file
    p_value_path = os.path.join(model_save_path, 'pvalues')
    if not os.path.exists(p_value_path):
        os.makedirs(p_value_path)
    np.save(os.path.join(p_value_path, 'drug{}.npy'.format(args.drug)), p_values)











