'''
Creates a binary matrix of biomarker features and runs a binary matrix
factorization routine on it. Code for factorizing the matrix is courtesy
of Jackson Loper. We use k=50 latent factors and run the model for 30 minutes
which appears to be sufficient (on a 2017 MacBook Pro) to converge.
'''
import pandas as pd
import numpy as np
import pickle

import asdf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
import lowlevel
import scipy.sparse.linalg
import numpy.random as npr
import scipy as sp
import tqdm
import threading
import scipy.stats 
import sys
import time

if __name__ == '__main__':
    np.random.seed(42)

    # Load the data
    filename = sys.argv[1]
    print('Loading data from {}'.format(filename))
    v=pd.read_csv(filename, header=0, index_col=0)

    print('Getting binary features')
    mut_cols = [c for c in v.columns if c.startswith('MUT')]
    X_mut = v[mut_cols].values

    # Collect the CNV columns and split into gain and loss
    cnv_cols = [c for c in v.columns if c.startswith('CNV')]
    X_cnv = v[cnv_cols].values
    X_cn_loss = (X_cnv < 1).astype(int)
    X_cn_gain = (X_cnv > 1).astype(int)


    # Collect the EXP columns and split into over and under expression
    exp_cols = [c for c in v.columns if c.startswith('EXP')]
    X_exp = v[exp_cols].values
    X_exp_under = (X_exp < (X_exp.mean(axis=0, keepdims=True) - X_exp.std(axis=0, keepdims=True)))
    X_exp_over = (X_exp > (X_exp.mean(axis=0, keepdims=True) + X_exp.std(axis=0, keepdims=True)))

    tis_cols = [c for c in v.columns if c.startswith('TISSUE')]
    X_tissue = v[tis_cols].values

    # Merge the new binarized values
    new_cols = (mut_cols
            + [c.replace('CNV', 'CNLoss') for c in cnv_cols]
            + [c.replace('CNV', 'CNGain') for c in cnv_cols]
            + [c.replace('EXP', 'EXPUnder') for c in exp_cols]
            + [c.replace('EXP', 'EXPOver') for c in exp_cols]
            + tis_cols)
    X_binarized = np.concatenate([X_mut, X_cn_loss, X_cn_gain, X_exp_under, X_exp_over, X_tissue], axis=1)

    nrows = X_binarized.shape[0]
    ncols = X_binarized.shape[1]
    print('Running binary matrix factorization with {} rows and {} columns'.format(nrows, ncols))

    data=asdf.BlockData(
        blocks=[
            asdf.Block('bernoulli',X_binarized.astype(float)),
        ],
        nrows=nrows,
        ncols=ncols,
    )

    # initialize
    Nk=50
    mod=asdf.initialize(data,Nk)
    snap=mod.snapshot()
    U=mod.row_loading
    V=mod.col_loading

    # start training in a background thread
    mod=asdf.Model.load(data,snap)

    if 'trainer' in locals():
        trainer.stop_thread()

    trainer=asdf.Trainer(mod)
    trainer.train_thread()

    # Wait for half an hour to give it time to train to convergence
    time.sleep(60*30)
    trainer.stop_thread()

    # save the results
    print('Saving to file')
    pd.DataFrame(X_binarized, columns=new_cols, index=v.index).to_csv(filename.replace('.csv', '_binarized.csv'))
    np.save(filename.replace('.csv', '_binarized_row_loading'), mod.row_loading)
    np.save(filename.replace('.csv', '_binarized_col_loading'), mod.col_loading)

