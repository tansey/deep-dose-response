import sys
import numpy as np
from scipy.stats import gamma
from utils import ilogit
from scipy.stats import invwishart, poisson, multivariate_normal as mvn
from elliptical_slice import elliptical_slice

def posterior_ess_helper(x):
    (y, present,
     grid, weights, c,
     mu, beta, chol,
     nburn, nsamples, nthin) = x
    
    def log_likelihood_fn(proposal_beta, dummy):
        if np.any(proposal_beta[:-1] > proposal_beta[1:]):
            return -np.inf
        tau = ilogit(proposal_beta)[present][:,np.newaxis]
        return np.log((poisson.pmf(y, grid * tau + c) * weights).clip(1e-10,np.inf).sum(axis=1)).sum()

    beta_samples = np.zeros((nsamples, mu.shape[0]))
    loglikelihood_samples = np.zeros(nsamples)
    for step in range(nburn+nsamples*nthin):
        # Ellipitical slice sample for beta
        cur_ll = None if step == 0 else cur_log_likelihood
        beta, cur_log_likelihood = elliptical_slice(beta, chol,
                                                      log_likelihood_fn,
                                                      cur_log_like=cur_ll,
                                                      ll_args=None,
                                                      mu=mu)

        # Save this sample after burn-in and markov chain thinning
        if step < nburn or ((step-nburn) % nthin) != 0:
            continue

        # Save the samples
        sample_idx = (step - nburn) // nthin
        beta_samples[sample_idx] = beta
        loglikelihood_samples[sample_idx] = cur_log_likelihood
    return beta_samples, loglikelihood_samples


def posterior_ess(Y, M, Sigma, A, B, C,
                            Beta=None,
                            lam_gridsize=100,
                            nburn=1000, nsamples=1000, nthin=1,
                            nthreads=1,
                            print_freq=100):
    # Filter out the unknown Y values
    Present = Y >= 0

    if Beta is None:
        # Initialize beta to the approximate MLE where data is not missing
        # and the prior where data is missing
        Beta = M*(1-Present) + Present*((Y - C[:,None]) / A[:,None] * B[:,None]).clip(1e-6, 1-1e-6)

    # Use a grid approximation for lambda integral
    Lam_grid, Lam_weights = [], []
    for a, b, c in zip(A, B, C):
        grid = np.linspace(gamma.ppf(1e-3, a, scale=b), gamma.ppf(1-1e-3, a, scale=b), lam_gridsize)[np.newaxis,:]
        weights = gamma.pdf(grid, a, scale=b)
        weights /= weights.sum()
        Lam_grid.append(grid)
        Lam_weights.append(weights)
    Lam_grid = np.array(Lam_grid)
    Lam_weights = np.array(Lam_weights)
    
    
    # Create the results arrays
    Cur_log_likelihood = np.zeros(M.shape[0])
    chol = np.linalg.cholesky(Sigma)
    Beta_samples = np.zeros((nsamples, Beta.shape[0], Beta.shape[1]))
    Loglikelihood_samples = np.zeros(nsamples)

    if nthreads == 1:
        ### Create a log-likelihood function for the ES sampler ###
        def log_likelihood_fn(proposal_beta, idx):
            if np.any(proposal_beta[:-1] > proposal_beta[1:]):
                return -np.inf
            present = Present[idx]
            y = Y[idx][present][:,np.newaxis]
            tau = ilogit(proposal_beta)[present][:,np.newaxis]
            grid = Lam_grid[idx]
            weights = Lam_weights[idx]
            c = C[idx]
            return np.log((poisson.pmf(y, grid * tau + c) * weights).clip(1e-10,np.inf).sum(axis=1)).sum()

        # Run the MCMC sampler on a single thread
        for step in range(nburn+nsamples*nthin):
            if print_freq and step % print_freq == 0:
                if step > 0:
                    sys.stdout.write("\033[F") # Cursor up one line
                print('MCMC step {}'.format(step))

            # Ellipitical slice sample for each beta
            for idx, beta in enumerate(Beta):
                cur_ll = None if step == 0 else Cur_log_likelihood[idx]
                Beta[idx], Cur_log_likelihood[idx] = elliptical_slice(beta, chol,
                                                                      log_likelihood_fn,
                                                                      cur_log_like=cur_ll,
                                                                      ll_args=idx,
                                                                      mu=M[idx])

            # Save this sample after burn-in and markov chain thinning
            if step < nburn or ((step-nburn) % nthin) != 0:
                continue

            # Save the samples
            sample_idx = (step - nburn) // nthin
            Beta_samples[sample_idx] = Beta
            Loglikelihood_samples[sample_idx] = Cur_log_likelihood.sum()
    else:
        from multiprocessing import Pool
        jobs = [(Y[idx][Present[idx]][:,np.newaxis],
                 Present[idx],
                 Lam_grid[idx],
                 Lam_weights[idx],
                 C[idx],
                 M[idx],
                 Beta[idx],
                 chol,
                 nburn, nsamples, nthin) for idx in range(Beta.shape[0])]
        
        # Calculate the posteriors in parallel
        with Pool(nthreads) as pool:
            results = pool.map(posterior_ess_helper, jobs)

            # Aggregate the results
            for idx in range(Beta.shape[0]):
                Beta_samples[:,idx] = results[idx][0]
                Loglikelihood_samples += results[idx][1]

    return Beta_samples, Loglikelihood_samples


def test_posterior_ess():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import monotone_rejection_sampler
    M = np.array([[-3, -2, -0.4, 0,   1, 1.1, 1.8, 3, 4],
                  [-7,-3, -0.1, 1.2, 1.5, 2.5, 3., 3.9, 4],
                  [-1,-0.5, 0, 1., 1.25, 2.5, 3.8, 3.9, 4]])
    N = M.shape[0]
    ndoses = M.shape[1]
    A, B, C = np.random.gamma(5, 10, size=N), np.random.gamma(1000, 10, size=N), np.random.gamma(10,10, size=N) # b is a scale param
    n_pos_ctrl = 40
    bandwidth, kernel_scale, noise_var = 1., 2., 0.05
    Sigma = np.array([kernel_scale*(np.exp(-0.5*(i - np.arange(ndoses))**2 / bandwidth**2)) for i in np.arange(ndoses)]) + noise_var*np.eye(ndoses) # squared exponential kernel
    Beta = np.array([monotone_rejection_sampler(m, Sigma) for m in M])
    Tau = ilogit(Beta)
    Lam_y = np.array([np.random.gamma(a, b, size=ndoses) for a, b in zip(A, B)])
    Lam_r = np.array([np.random.gamma(a, b, size=n_pos_ctrl) for a, b in zip(A, B)])
    Y = np.random.poisson(Tau * Lam_y + C[:,np.newaxis])
    R = np.random.poisson(Lam_r + C[:,np.newaxis])

    # Add some missing dosages to predict
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.random.random() < 0.1:
                Y[i,j] = -1

    colors = ['blue', 'orange', 'green']
    [plt.plot(t, color=color) for t,color in zip(Tau, colors)]
    [plt.scatter(np.arange(M.shape[1])[y >= 0], ((y[y >= 0] - c) / (r.mean() - c)).clip(0,1), color=color) for y, r, c, color in zip(Y, R, C, colors)]
    plt.show()
    plt.close()

    Beta_hat = posterior_ess(Y, M, Sigma, A, B, C)
    Tau_hat = ilogit(Beta_hat)

    Beta_hat2 = posterior_ess(Y, M, Sigma, A, B, C)
    Tau_hat2 = ilogit(Beta_hat2)

    Beta_hat3 = posterior_ess(Y, M, Sigma, A, B, C)
    Tau_hat3 = ilogit(Beta_hat3)

    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=2)
        plt.rc('axes', lw=2)

        colors = ['blue', 'orange', 'green']
        fig, axarr = plt.subplots(1,3, sharex=True, sharey=True)
        for ax, y, t, t_hat, t_hat2, t_hat3, t_lower, t_upper, r, c, color in zip(axarr, Y, Tau,
                                                            Tau_hat.mean(axis=0),
                                                            Tau_hat2.mean(axis=0),
                                                            Tau_hat3.mean(axis=0),
                                                            np.percentile(Tau_hat, 5, axis=0),
                                                            np.percentile(Tau_hat, 95, axis=0),
                                                            R, C,
                                                            colors):
            ax.scatter(np.arange(M.shape[1])[y >= 0], ((y[y >= 0] - c) / (r.mean() - c)).clip(0,1), color=color)
            ax.plot(np.arange(M.shape[1]), t, color=color, lw=3, ls='--')
            ax.plot(np.arange(M.shape[1]), t_hat, color=color, lw=3)
            ax.plot(np.arange(M.shape[1]), t_hat2, color=color, lw=3)
            ax.plot(np.arange(M.shape[1]), t_hat3, color=color, lw=3)
            ax.fill_between(np.arange(M.shape[1]), t_lower, t_upper, color=color, alpha=0.5)
            ax.set_xlabel('Dosage level', fontsize=18, weight='bold')
            ax.set_ylabel('Survival percentage', fontsize=18, weight='bold')
    plt.show()

def posterior_ess_Sigma(Y, M, A, B, C,
                            Sigma=None, nu=None, Psi=None, Beta=None,
                            lam_gridsize=100,
                            nburn=500, nsamples=1000, nthin=1,
                            print_freq=100):
    if nu is None:
        # Default degrees of freedom
        nu = M.shape[1]+1

    if Psi is None:
        # # Default squared exponential kernel prior
        # bandwidth, kernel_scale, noise_var = 2., 1., 0.5
        # Psi = np.array([kernel_scale*(np.exp(-0.5*(i - np.arange(M.shape[1]))**2 / bandwidth**2)) for i in np.arange(M.shape[1])]) + noise_var*np.eye(M.shape[1])
        Psi = np.eye(M.shape[1])
        Psi *= nu - M.shape[1] + 1

    if Sigma is None:
        # Sample from the prior to initialize Sigma
        Sigma = invwishart.rvs(nu, Psi)

    if Beta is None:
        Beta = np.copy(M)

    # Filter out the unknown Y values
    Present = Y >= 0

    # Use a grid approximation for lambda integral
    Lam_grid, Lam_weights = [], []
    for a, b, c in zip(A, B, C):
        grid = np.linspace(gamma.ppf(1e-3, a, scale=b), gamma.ppf(1-1e-3, a, scale=b), lam_gridsize)[np.newaxis,:]
        weights = gamma.pdf(grid, a, scale=b)
        weights /= weights.sum()
        Lam_grid.append(grid)
        Lam_weights.append(weights)
    Lam_grid = np.array(Lam_grid)
    Lam_weights = np.array(Lam_weights)
    
    ### Create a log-likelihood function for the ES sampler ###
    def log_likelihood_fn(proposal_beta, idx):
        if np.any(proposal_beta[:-1] > proposal_beta[1:]+1e-6):
            return -np.inf
        present = Present[idx]
        y = Y[idx][present][:,np.newaxis]
        tau = ilogit(proposal_beta)[present][:,np.newaxis]
        grid = Lam_grid[idx]
        weights = Lam_weights[idx]
        c = C[idx]
        return np.log((poisson.pmf(y, grid * tau + c) * weights).clip(1e-10,np.inf).sum(axis=1)).sum()

    # Initialize betas with draws from the prior
    Cur_log_likelihood = np.zeros(M.shape[0])
    chol = np.linalg.cholesky(Sigma)

    # Create the results arrays
    Beta_samples = np.zeros((nsamples, Beta.shape[0], Beta.shape[1]))
    Sigma_samples = np.zeros((nsamples, Sigma.shape[0], Sigma.shape[1]))
    Loglikelihood_samples = np.zeros(nsamples)

    # Run the MCMC sampler
    for step in range(nburn+nsamples*nthin):
        if print_freq and step % print_freq == 0:
            if step > 0:
                sys.stdout.write("\033[F") # Cursor up one line
            print('MCMC step {}'.format(step))

        # Ellipitical slice sample for each beta
        for idx, beta in enumerate(Beta):
            Beta[idx], Cur_log_likelihood[idx] = elliptical_slice(beta, chol,
                                                                  log_likelihood_fn,
                                                                  ll_args=idx,
                                                                  mu=M[idx])
            # Cur_log_likelihood[idx] += mvn.logpdf(Beta[idx], M[idx], Sigma)

        # Conjugate prior update for Sigma
        Sigma = invwishart.rvs(nu+M.shape[0], Psi+(Beta - M).T.dot(Beta - M))

        # Cholesky representation
        chol = np.linalg.cholesky(Sigma)

        # Save this sample after burn-in and markov chain thinning
        if step < nburn or ((step-nburn) % nthin) != 0:
            continue

        # Save the samples
        sample_idx = (step - nburn) // nthin
        Beta_samples[sample_idx] = Beta
        Sigma_samples[sample_idx] = Sigma
        Loglikelihood_samples[sample_idx] = Cur_log_likelihood.sum()

    return Beta_samples, Sigma_samples, Loglikelihood_samples


def test_ess_Sigma():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import monotone_rejection_sampler
    N = 100
    ndoses = 9
    M = np.random.normal(0,4,size=(N,ndoses))
    M.sort(axis=1)
    A, B, C = np.random.gamma(5, 10, size=N), np.random.gamma(1000, 10, size=N), np.random.gamma(10,10, size=N) # b is a scale param
    n_pos_ctrl = 40
    bandwidth, kernel_scale, noise_var = 1., 2., 0.05
    Sigma = np.array([kernel_scale*(np.exp(-0.5*(i - np.arange(ndoses))**2 / bandwidth**2)) for i in np.arange(ndoses)]) + noise_var*np.eye(ndoses) # squared exponential kernel
    Beta = np.array([monotone_rejection_sampler(m, Sigma) for m in M])
    Tau = ilogit(Beta)
    Lam_y = np.array([np.random.gamma(a, b, size=ndoses) for a, b in zip(A, B)])
    Lam_r = np.array([np.random.gamma(a, b, size=n_pos_ctrl) for a, b in zip(A, B)])
    Y = np.random.poisson(Tau * Lam_y + C[:,np.newaxis])
    R = np.random.poisson(Lam_r + C[:,np.newaxis])

    # Add some missing dosages to predict
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.random.random() < 0.1:
                Y[i,j] = -1

    colors = ['blue', 'orange', 'green']
    [plt.plot(t, color=color) for t,color in zip(Tau, colors)]
    [plt.scatter(np.arange(ndoses)[y >= 0], ((y[y >= 0] - c) / (r.mean() - c)).clip(0,1), color=color) for y, c, r, color in zip(Y, C, R, colors)]
    plt.show()
    plt.close()

    Beta_samples, Sigma_samples, Loglikelihood_samples = posterior_ess_Sigma(Y, M, A, B, C, Sigma=Sigma)

    from utils import pretty_str
    print('Truth:')
    print(pretty_str(Sigma))
    print('')
    print('Bayes estimate:')
    print(pretty_str(Sigma_samples.mean(axis=0)))
    print('Last sample:')
    print(pretty_str(Sigma_samples[-1]))


