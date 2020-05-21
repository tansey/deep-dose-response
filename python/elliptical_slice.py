"""
ELLIPTICAL_SLICE Markov chain update for a distribution with a Gaussian "prior" factored out

    [xx, cur_log_like] = elliptical_slice(xx, chol_Sigma, log_like_fn);
OR
    [xx, cur_log_like] = elliptical_slice(xx, prior_sample, log_like_fn);

Optional additional arguments: cur_log_like, angle_range, varargin (see below).

A Markov chain update is applied to the D-element array xx leaving a
"posterior" distribution
    P(xx) \propto N(xx;0,Sigma) L(xx)
invariant. Where N(0,Sigma) is a zero-mean Gaussian distribution with
covariance Sigma. Often L is a likelihood function in an inference problem.

Inputs:
             xx Dx1 initial vector (can be any array with D elements)

     chol_Sigma DxD chol(Sigma). Sigma is the prior covariance of xx
 or:
   prior_sample Dx1 single sample from N(0, Sigma)

    log_like_fn @fn log_like_fn(xx, varargin{:}) returns 1x1 log likelihood

Optional inputs:
   cur_log_like 1x1 log_like_fn(xx, varargin{:}) of initial vector.
                    You can omit this argument or pass [].
    angle_range 1x1 Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
                    You can omit this argument or pass [].
       varargin  -  any additional arguments are passed to log_like_fn

Outputs:
             xx Dx1 (size matches input) perturbed vector
   cur_log_like 1x1 log_like_fn(xx, varargin{:}) of final vector

Iain Murray, September 2009
Tweak to interface and documentation, September 2010
Updated by Scott Linderman, 2013-2014

Reference:
Elliptical slice sampling
Iain Murray, Ryan Prescott Adams and David J.C. MacKay.
The Proceedings of the 13th International Conference on Artificial
Intelligence and Statistics (AISTATS), JMLR W&CP 9:541-548, 2010.

"""
import math
import numpy as np


def elliptical_slice(xx, prior, log_like_fn, cur_log_like=None, angle_range=0, ll_args=None, mu=None):
    # Only work with a copy of xx
    xx = np.copy(xx)
    
    D = np.size(xx)
    
    if np.size(prior) == D:
        # User provided a prior sample
        nu = np.reshape(prior, (D,))
    else:
        # User provided Cholesky of prior covariance
        if np.shape(prior) != (D,D):
            raise Exception("Prior must be given by a D-element sample or DxD chol(Sigma, 'lower')")
            
        nu = np.reshape(np.dot(prior, np.random.randn(D,1)).T, np.shape(xx))
        
    if mu is None:
        mu = np.zeros(D)
    elif np.size(mu)!=D:
        raise Exception("Specified mean does not have the correct shape!")

    if (cur_log_like is None):
        cur_log_like = log_like_fn(xx, ll_args)
    
    init_ll = cur_log_like
    hh = np.log(np.random.rand()) + cur_log_like
    
    # Set up the bracket of angles and pick first proposal
    # phi = (theta' - theta) is a change in angle
    if angle_range <= 0:
        phi = np.random.rand() * 2 * math.pi
        phi_min = phi - 2*math.pi
        phi_max = phi
    else:
        phi_min = -1 * angle_range * np.random.rand()
        phi_max = phi_min + angle_range
        phi = np.random.rand() * (phi_max - phi_min) + phi_min
        
    # Slice sampling loop
    while True:
        # compute xx for proposed angle difference and check if it's on the slice
        # Add the offset mu before computing the likelihood
        xx_prop = ((xx-mu)*np.cos(phi) + nu*np.sin(phi)) + mu
        cur_log_like = log_like_fn(xx_prop, ll_args)
        if cur_log_like >= hh:
            # New point is on the slice -> Exit loop
            break
        
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            import warnings
            assert np.allclose(xx, xx_prop)
            warnings.warn("Possible BUG: Shrunk to current position and still rejected!")
            break
            
        phi = np.random.rand()*(phi_max - phi_min) + phi_min
        
    return xx_prop, cur_log_like
            
                
                