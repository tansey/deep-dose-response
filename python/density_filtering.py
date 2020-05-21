'''Example showing how to use trend filtering to perform density regression in
a time series model.'''
import numpy as np
from scipy.stats import norm

def get_1d_penalty_matrix(x, k=0, sparse=False):
    '''Create a 1D trend filtering penalty matrix D^(k+1).'''
    length = len(x)
    if sparse:
        rows = np.repeat(np.arange(length-1), 2)
        cols = np.repeat(np.arange(length), 2)[1:-1]
        data = np.tile([-1, 1], length-1)
        D = coo_matrix((data, (rows, cols)), shape=(length-1, length))
    else:
        D = np.eye(length, dtype=float)[0:-1] * -1
        for i in range(len(D)):
            D[i,i+1] = 1
    return get_delta(D, k, x)

def get_delta(D, k, x=None):
    '''Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k. If x is specified, then we use the
    falling factorial basis from Wang et al. (ICML 2014) to specifiy an
    irregular grid.'''
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = D
    for i in range(k):
        if x is not None:
            z = i+1
            W = np.diag(float(z) / (x[z:] - x[:-z]))
            result = D.T.dot(W).dot(result) if i % 2 == 0 else D.dot(W).dot(result)
        else:
            result = D.T.dot(result) if i % 2 == 0 else D.dot(result)
    return result

def sufficient_statistics(x, y):
    T = []
    X = []
    W = []
    for i in range(x.min(), x.max()+1):
        xi = x[x==i]
        if len(xi) > 0:
            yi = y[x==i]
            t1 = np.sum(yi)
            t2 = np.sum(yi**2)
            X.append(i)
            W.append(len(xi))
            T.append([t1,t2])
    return np.array(T), np.array(X), np.array(W)

def tf_fit(T, X, W, D0, D1, lam1, lam2, 
            init_eta1=None,
            init_eta2_raw=None,
            nsteps=10000,
            learning_rate_fn=lambda s: 0.995**(s+1),
            verbose=True):
    import tensorflow as tf
    # Setup the TF model
    tf.reset_default_graph()
    tf_sess = tf.Session()

    if init_eta1 is None:
        init_eta1 = np.ones_like(T[:,0]).astype('float32')
    if init_eta2_raw is None:
        init_eta2_raw = np.ones_like(T[:,1]).astype('float32')

    # Create the data tensors
    tf_W = tf.constant(W, tf.float32)
    tf_T = tf.constant(T, tf.float32)
    tf_D0 = tf.constant(D0, tf.float32)
    tf_D1 = tf.constant(D1, tf.float32)

    # Work in natural parameter space
    tf_eta1 = tf.get_variable('Eta1', initializer=init_eta1)
    tf_eta2_raw = tf.get_variable('Eta2', initializer=init_eta2_raw)
    tf_eta2 = -tf.nn.softplus(tf_eta2_raw)
    tf_eta = tf.stack([tf_eta1, tf_eta2], axis=1)
    tf_mean, tf_variance = tf_eta1 / (-2 * tf_eta2), 1. / (-2 * tf_eta2)
    
    # Use a gaussian loss
    log_kernel = tf.reduce_sum(tf_T * tf_eta, 1)
    log_partition = tf_eta1**2 / (4.*tf_eta2) + 0.5 * tf.log(-2 * tf_eta2)
    
    # Use a model that has piecewise-linear means and constant variance
    mean_penalty = tf.reduce_sum(tf.abs(tf.matmul(tf_D1, tf_mean[:,None])))
    var_penalty = tf.reduce_sum(tf.abs(tf.matmul(tf_D0, tf_variance[:,None])))

    # Use a model that has piecewise-linear first natural parameter and
    # piecewise-constant second natural parameter
    # mean_penalty = tf.reduce_sum(tf.abs(tf.matmul(tf_D1, tf_eta1[:,None])))
    # var_penalty = tf.reduce_sum(tf.abs(tf.matmul(tf_D0, tf_eta2[:,None])))

    # Get the convex optimization loss
    loss = -tf.reduce_sum(log_kernel + W * log_partition) + lam1 * mean_penalty + lam2 * var_penalty

    # Setup optimizer
    tf_learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(tf_learning_rate)
    tf_train_step = opt.minimize(loss)

    # Fit the model
    tf_sess.run(tf.global_variables_initializer())
    for step in range(nsteps):
        if verbose and (step % 1000) == 0:
            # Convert back to normal parameter space
            mean, variance = tf_sess.run([tf_eta1 / (-2 * tf_eta2), 1. / (-2 * tf_eta2)])
            eta = tf_sess.run(tf_eta[0])
            loss_kernel, loss_part, loss_mean, loss_var = tf_sess.run([tf.reduce_sum(log_kernel), tf.reduce_sum(W*log_partition), mean_penalty, var_penalty])

            print('\n\n********** STEP {} **********'.format(step))
            print('Data.\n\tT: {}\n\tW: {}\n\tX: {}'.format(T[0], W[0], X[0]))
            print('Parameters.\n\tMean: {}\n\tVariance: {}\n\tEta: {}'.format(mean[0], variance[0], eta))
            print('Loss components.\n\tKernel: {}\n\tParition: {}\n\tMean: {}\n\tVariance: {}'.format(loss_kernel, loss_part, loss_mean, loss_var) )
            print('Step size: {}'.format(learning_rate_fn(step)))

        tf_sess.run(tf_train_step, feed_dict={tf_learning_rate: learning_rate_fn(step)})

    # Convert back to normal parameter space
    return tf_sess.run([tf_mean, tf_variance])

def density_regression(x, y, lam1=10., lam2=5., 
                        init_eta1=None, init_eta2_raw=None,
                        nsteps=10000, verbose=True):
    # Convert to z-scores
    y_mu = y.mean()
    y_stdev = y.std()
    y = (y - y_mu) / y_stdev
    
    # Calculate the sufficient statistics under a normal likelihood
    T, X, W = sufficient_statistics(x, y)

    # Create the trend filtering penalty matrices
    D0 = get_1d_penalty_matrix(X, k=0, sparse=False)
    D1 = get_1d_penalty_matrix(X, k=1, sparse=False)

    # Fit the data under a normal distribution assumption
    fit_means, fit_variances = tf_fit(T, X, W, D0, D1, lam1, lam2,
                                learning_rate_fn=lambda s: 0.01,
                                nsteps=nsteps,
                                init_eta1=init_eta1,
                                init_eta2_raw=init_eta2_raw,
                                verbose=verbose)

    # Convert back from z-scores to raw values
    return X, fit_means * y_stdev + y_mu, fit_variances * y_stdev**2

def logspace_grid(min_val, max_val, npoints):
    return np.exp(np.linspace(np.log(min_val), np.log(max_val), npoints))

def create_folds(n, k):
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def predict(x, fit):
    fit_x, means, variances = fit
    pred_mean = np.interp(x, fit_x, means)
    pred_var = np.interp(x, fit_x, variances)
    return pred_mean, pred_var

def logprob(x, y, fit):
    pred_mean, pred_var = predict(x, fit)
    logprobs = norm.logpdf(y, pred_mean, np.sqrt(pred_var))
    return logprobs

def density_regression_cv(x, y, nfolds=5,
                          min_lam1=1e-2, max_lam1=2e2, nlam1=10,
                          min_lam2=1e-2, max_lam2=2e2, nlam2=10):
    '''Cross-validation to select the value of lambda 1 and lambda 2 based on
    log probability.'''
    lam1_grid = logspace_grid(min_lam1, max_lam1, nlam1)
    lam2_grid = logspace_grid(min_lam2, max_lam2, nlam2)
    folds = create_folds(len(x), nfolds)
    cv_scores = np.zeros((nlam1, nlam2))
    for fold_num, fold in enumerate(folds):
        print('\tFold #{0}'.format(fold_num+1))
        mask = np.ones(len(x), dtype=bool)
        mask[fold] = False
        x_train, y_train = x[mask], y[mask]
        x_test, y_test = x[~mask], y[~mask]
        prev_init_eta1, prev_init_eta2_raw = None, None
        for i,lam1 in enumerate(lam1_grid):
            init_eta1 = prev_init_eta1
            init_eta2_raw = prev_init_eta2_raw
            for j,lam2 in enumerate(lam2_grid):
                print('\n\t\tlam1={} lam2={}'.format(lam1, lam2))
                fold_x, fold_means, fold_variances = density_regression(x_train, y_train, lam1, lam2, 
                                                        init_eta1=init_eta1,
                                                        init_eta2_raw=init_eta2_raw,
                                                        verbose=False, nsteps=3000)
                score = logprob(x_test, y_test, (fold_x, fold_means, fold_variances)).sum()
                print('\t\t\t score={}'.format(score))
                cv_scores[i,j] += score

                init_eta1 = fold_means / fold_variances
                init_eta2_raw = np.exp(1./(2*fold_variances) - 1.)
                if j == 0 and not np.isnan(score):
                    prev_init_eta1 = np.copy(init_eta1)
                    prev_init_eta2_raw = np.copy(init_eta2_raw)
    best_idx = np.nanargmax(cv_scores)
    best_lam1 = lam1_grid[int(np.floor(best_idx // nlam2))]
    best_lam2 = lam2_grid[int(best_idx % nlam2)]
    fit = density_regression(x, y, best_lam1, best_lam2, nsteps=20000, verbose=False)
    print(best_lam1, best_lam2)
    print('Best selected values: lam1={} lam2={}'.format(best_lam1, best_lam2))
    return fit, (best_lam1, best_lam2)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    n = 200
    m = 8
    offset = 7000.
    p = np.array([0.3]*(n/5) + [0.01]*(n/10) + [0.9]*(n/10) + [0.3]*(n/4) + [0.5]*(n/4) + [0.5]*(n/10))
    slopes = np.array([50.]*(n/4) + [-30]*(n/4) + [-20]*(n/4) + [40]*(n/4))
    variances = np.array([100000.]*(n/4) + [500000.]*(n/4) + [10000.]*(n/4) + [200000.]*(n/4))
    means = np.zeros(n)
    means[0] = offset

    assert len(p) == n
    assert len(slopes) == n
    assert len(variances) == n

    # Generate the data
    m0 = np.random.poisson(m)
    x = [0]*m0
    y = list(np.random.normal(means[0], np.sqrt(variances[0]), size=m0))
    for i in range(1,n):
        means[i] = means[i-1] + slopes[i]
        if p[i] < np.random.random():
            nsamples = np.random.poisson(m)
            if nsamples > 0:
                x.extend([i] * nsamples)
                y.extend(np.random.normal(means[i], np.sqrt(variances[i]), size=nsamples))
    x = np.array(x)
    y = np.array(y)

    # Fit the model by 5-fold cross-validation
    # fit_x, fit_means, fit_variances = density_regression_cv(x, y)

    # Use some decent values found via cross-validation
    fit_x, fit_means, fit_variances = density_regression(x, y, lam1=3.5, lam2=20.)
    
    # Plot a comparison of the truth vs. the fit
    fig, axarr = plt.subplots(1,2, sharex=True, sharey=True)
    axarr[0].scatter(x, y, alpha=0.7, color='gray')
    axarr[0].plot(np.arange(n), means, label='Truth')
    axarr[0].fill_between(np.arange(n), means + 2*np.sqrt(variances), means - 2*np.sqrt(variances), color='blue', alpha=0.3)
    axarr[0].set_title('Truth')

    axarr[1].scatter(x, y, alpha=0.7, color='gray')
    axarr[1].plot(fit_x, fit_means, label='Truth')
    axarr[1].fill_between(fit_x, fit_means+2*np.sqrt(fit_variances), fit_means-2*np.sqrt(fit_variances), color='blue', alpha=0.3)
    axarr[1].set_title('Fit')
    plt.show()


