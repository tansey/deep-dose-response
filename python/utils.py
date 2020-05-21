import numpy as np
import pandas as pd
import scipy.stats as st
import os

def try_load(filename, load_fn, fail=None, warning=None):
    if os.path.exists(filename):
        return load_fn(filename)
    if warning is not None:
        import warnings
        warnings.warn(warning)
    if fail is not None:
        return fail(filename)
    return None

def load_dataset(infile, index_col=None, delimiter=None):
    # Handle reading in xlsx or csv files
    if infile.endswith('xlsx'):
        return pd.read_excel(infile, header=0, index_col=index_col)
    return pd.read_csv(infile, header=0, index_col=index_col, delimiter=delimiter)

def pretty_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, ignore)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, ignore, label_columns)
    if len(p.shape) > 2:
        return '[' + ',\n'.join([pretty_str(pi) for pi in p]) + ']'
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, ignore) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, ignore=None):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if ((hasattr(ignore, "__len__") and a in ignore) or a == ignore) else style.format(a) for a in p]))

def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def cols_as_np(batch, cols):
    # Convert to numpy array
    ctrls = batch[cols].values
    ctrls[~np.frompyfunc(isnumeric, 1,1)(ctrls).astype(bool)] = np.nan
    ctrls = ctrls.astype(float)
    return ctrls

def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def ilogit(x):
    return 1. / (1+np.exp(-x))

def pav(y):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of y

    translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    Author : Alexandre Gramfort
    license : BSD
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v


def benjamini_hochberg(p, fdr):
    '''Performs Benjamini-Hochberg multiple hypothesis testing on z at the given false discovery rate threshold.'''
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    discoveries = np.array(discoveries, dtype=int)
    return discoveries

def benjamini_hochberg_predictions(p, fdr_threshold):
    if type(p) is np.ndarray:
        pshape = p.shape
        if len(pshape) > 1:
            p = p.flatten()
    bh_discoveries = benjamini_hochberg(p, fdr_threshold)
    bh_predictions = np.zeros(len(p), dtype=int)
    if len(bh_discoveries) > 0:
        bh_predictions[bh_discoveries] = 1
    if type(p) is np.ndarray and len(pshape) > 1:
        bh_predictions = bh_predictions.reshape(pshape)
    return bh_predictions


def calc_p_value(z, mu0=0., sigma0=1., side='both'):
    import scipy.stats as st
    if side == 'both':
        return 2*(1.0 - st.norm.cdf(np.abs((z - mu0) / sigma0)))
    elif side == 'left':
        return st.norm.cdf((z - mu0) / sigma0)
    elif side == 'right':
        return 1 - st.norm.cdf((z - mu0) / sigma0)
    raise Exception('side must be either both, left, or right.')


def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b*batch_size:min((b+1)*batch_size, len(order))]
        yield idx


def true_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==1)).sum(axis=axis)

def false_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==0)).sum(axis=axis)

def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries, dtype=int)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def stable_softmax(x, axis=None):
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator/denominator
    return softmax
    

def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def create_matrix_folds(X, k):
    from itertools import product
    row_folds = create_folds(X.shape[0], k)
    col_folds = create_folds(X.shape[1], k)
    folds = [np.array(list(product(r,c))).T for r, c in zip(row_folds, col_folds)]
    return folds
    

def cv_kde_cdf(y, obs, b=None, min_b=1e-3, max_b=1e3, num_b=100, num_x=1000, num_folds=5):
    '''Select the KDE bandwidth via leave-one-out cross-validation.'''
    min_x, max_x = min(y.min(), obs.min()) - obs.std()*1, max(y.max(), obs.max()) + obs.std()*1
    x_grid = np.linspace(min_x, max_x, num_x)
    if b is None:
        norms = -0.5*(x_grid[:,np.newaxis] - obs[np.newaxis,:])**2
        b_grid = np.exp(np.linspace(np.log(min_b), np.log(max_b), num_b))
        results = np.zeros(len(b_grid))
        folds = create_folds(obs, num_folds)
        for i,fold in enumerate(folds):
            print('\tFold #{0}'.format(i))
            mask = np.ones(len(obs), dtype=bool)
            mask[fold] = False
            obs_train = obs[mask]
            obs_test = obs[~mask]
            norms_train = norms[:,mask]
            for i, b in enumerate(b_grid):
                print('\t{}'.format(i))
                weights = np.exp(norms_train/b**2).sum(axis=1)
                weights /= weights.sum()
                # print weights
                test_idx = np.searchsorted(x_grid, obs_test)
                right_weight = (obs_test - x_grid[test_idx - 1]) / (x_grid[test_idx] - x_grid[test_idx-1])
                test_probs = weights[test_idx - 1] * (1-right_weight) + weights[test_idx] * right_weight
                results[i] += np.log(test_probs).sum()
        results[np.isnan(results)] = -np.inf
        for i,r in enumerate(results):
            print(b_grid[i], r)
        b = b_grid[np.argmax(results)]
    print('b: {}'.format(b))
    weights = np.exp(-((x_grid[:,np.newaxis] - obs[np.newaxis,:])**2)/(2*b**2)).sum(axis=1)
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    indices = np.searchsorted(x_grid, y)
    right_weight = (y - x_grid[indices-1]) / (x_grid[indices] - x_grid[indices-1])
    return cdf[indices - 1] * (1-right_weight) + cdf[indices] * right_weight


def calc_fdr(probs, fdr_level):
    '''Calculates the detected signals at a specific false discovery rate given the posterior probabilities of each point.'''
    pshape = probs.shape
    if len(probs.shape) > 1:
        probs = probs.flatten()
    post_orders = np.argsort(probs)[::-1]
    avg_fdr = 0.0
    end_fdr = 0
    
    for idx in post_orders:
        test_fdr = (avg_fdr * end_fdr + (1.0 - probs[idx])) / (end_fdr + 1.0)
        if test_fdr > fdr_level:
            break
        avg_fdr = test_fdr
        end_fdr += 1

    is_finding = np.zeros(probs.shape, dtype=int)
    is_finding[post_orders[0:end_fdr]] = 1
    if len(pshape) > 1:
        is_finding = is_finding.reshape(pshape)
    return is_finding

def nb_fit(x, w=None, max_em_steps=30, min_r = 1e-6, min_p = 1e-6):
    '''Fit a negative binomial via EM, potentially with weighted observations.'''
    from scipy.special import gammaln, digamma, polygamma
    if w is None:
        w = np.array(1)
    wx_sum = (w*x).sum()
    w_sum = w.sum()
    wx_mean = wx_sum / w_sum
    wx_var = ((w * x - wx_mean)**2).sum() / w_sum
    p = (1. - wx_mean / wx_var).clip(0.01,0.99)
    r = wx_mean * (1-p) / p
    line_search_vals = np.exp(np.linspace(np.log(1e-12), 0., 20))
    for step in range(max_em_steps):
        # E-step fits p
        p = (wx_sum / (w_sum * r + wx_sum)).clip(min_p, 1.-min_p)
        # M-step fits r
        r_delta = 1.
        while r_delta > 1e-6:
            grad = -(w * digamma(x + r)).sum() + w_sum * digamma(r) - w_sum * np.log(1-p)
            hess = -(w * polygamma(1, x + r)).sum() + w_sum * polygamma(1, r)
            r_prev = r
            rvals = r - line_search_vals * grad / hess
            if np.isnan(rvals).sum() > 0:
                # Numerical stability check to see if the hessian has collapsed to zero
                break
            evals = np.array([np.inf] * len(line_search_vals))
            for i in range(len(line_search_vals)):
                if rvals[i] <= 0:
                    break
                evals[i] = -(w * gammaln(x + rvals[i])).sum() + w_sum * gammaln(rvals[i]) - wx_sum*np.log(p) - w_sum*rvals[i]*np.log(1-p)
            r = max(min_r, rvals[np.argmin(evals)])
            r_delta = np.abs(r - r_prev)
    return (r, p)


def negBinomRatio(k, r1, r2, p1, p2, log=False):
    from scipy.special import gammaln
    x = gammaln(k+r1) - gammaln(r1) - gammaln(k+r2) + gammaln(r2) + r1*np.log(1-p1) - r2*np.log(1-p2) + k * (np.log(p1) - np.log(p2))
    return x if log else np.exp(x)

def nb_fit_bayes(Z):
    from pypolyagamma import PyPolyaGamma
    from scipy.stats import norm
    results = []
    pgr = PyPolyaGamma(seed=0)
    model_logr = np.zeros(Z.shape[0])
    model_Psi = np.zeros(Z.shape)
    model_r = np.exp(model_logr)
    model_P = ilogit(model_Psi)
    prior_logr_sd = 100.
    Omegas = np.zeros_like(Z)
    for step in xrange(3000):
        # Random-walk MCMC for log(r)
        for mcmc_step in xrange(30):
            candidate_logr = model_logr + np.random.normal(0, 1, size=Z.shape[0])
            candidate_r = np.exp(candidate_logr)
            accept_prior = norm.logpdf(candidate_logr, loc=0, scale=prior_logr_sd) - norm.logpdf(model_logr, loc=0, scale=prior_logr_sd)
            accept_likelihood = negBinomRatio(Z, candidate_r[:,np.newaxis], model_r[:,np.newaxis], model_P, model_P, log=True).sum(axis=1)
            accept_probs = np.exp(np.clip(accept_prior + accept_likelihood, -10, 1))
            accept_indices = np.random.random(size=Z.shape[0]) <= accept_probs
            model_logr[accept_indices] = candidate_logr[accept_indices]
            model_r = np.exp(model_logr)
        
        # Polya-Gamma sampler -- Marginal test version only
        N_ij = Z + model_r[:,np.newaxis]
        [pgr.pgdrawv(N_ij[i], np.repeat(model_Psi[i,0], Z.shape[1]), Omegas[i]) for i in xrange(Z.shape[0])]

        # Sample the logits using only the expressed values -- Marginal test version only
        v = 1 / (Omegas.sum(axis=1) + 1/100.**2)
        m = v * (Z.sum(axis=1) - Z.shape[1] * model_r) / 2.
        model_Psi = np.random.normal(loc=m, scale=np.sqrt(v))[:,np.newaxis]
        model_P = ilogit(model_Psi)

        if step > 1000 and (step % 2) == 0:
            results.append([model_r, model_P[:,0]])
            # print(model_r, model_P[:,0])
    return np.array(results)

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

    Taken from https://github.com/pytorch/pytorch/issues/2591
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    import torch
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def clip_gradient(model, clip=5):
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.clamp(-clip,clip)

def knockoff_filter(knockoff_stats, alpha, offset=1.0, is_sorted=False):
    '''Perform the knockoffs selection procedure at the target FDR threshold.
    Note that this implementation runs in nlogn time.'''
    n = len(knockoff_stats) # Length of the stats array
    if is_sorted:
        order = np.arange(n)
        sorted_stats = knockoff_stats
    else:
        order = np.argsort(knockoff_stats)
        sorted_stats = knockoff_stats[order]

    # Edge case: if there are no positive stats, just return empty selections
    if sorted_stats[-1] <= 0:
        return np.array([], dtype='int64')
    
    ridx = np.searchsorted(sorted_stats, 0, side='right') # find smallest positive value index
    lidx = np.searchsorted(sorted_stats, -sorted_stats[ridx], side='left') # find matching negative value index
    
    # Numpy correction: if -sorted_stats[ridx] is less than any number in the list,
    # searchsorted returns 0 instead of -1. This is not the desired behavior here.
    if lidx == 0 and sorted_stats[lidx] >= -sorted_stats[ridx]:
        # If the current ratio isn't good enough, it will never get better.
        if (lidx + 1 + offset) / max(1, n - ridx) > alpha:
            return np.array([], dtype='int64')

        # If we're below the alpha threshold, return everything positive
        return order[ridx:]
    
    # If the knockoff ratio is below the threshold, return all stats
    # at or above the current value
    while (lidx + 1 + offset) / max(1, n - ridx) > alpha:
        # If we were at the end of the negative values, we won't get
        # a better ratio by going further down the positive value side.
        if lidx == -1:
            return np.array([], dtype='int64')

        # Move to the next smallest positive value
        ridx += 1

        # Check if we've reached the end of the list
        if ridx == n:
            break

        # Find the matching negative value
        while lidx >= 0 and sorted_stats[lidx] > -sorted_stats[ridx]:
            lidx -= 1

    # Return the set of stats with values above the threshold
    print('found: {}'.format(len(order) - ridx))
    return order[ridx:]



def monotone_rejection_sampler(m, Sigma):
    beta = np.random.multivariate_normal(m, Sigma)
    while np.any(beta[:-1] > beta[1:]):
        beta = np.random.multivariate_normal(m, Sigma)
    return beta

GDSC_DRUG_NAMES = {
    1:   'Erlotinib',
    3:   'Rapamycin',
    5:   'Sunitinib',
    6:   'PHA-665752',
    9:   'MG-132',
    11:  'Paclitaxel',
    17:  'Cyclopamine',
    29:  'AZ628',
    30:  'Sorafenib',
    32:  'Tozasertib',
    34:  'Imatinib',
    35:  'NVP-TAE684',
    37:  'Crizotinib',
    38:  'Saracatinib',
    41:  'S-Trityl-L-cysteine',
    45:  'Z-LLNle-CHO',
    51:  'Dasatinib',
    52:  'GNF-2',
    53:  'CGP-60474',
    54:  'CGP-082996',
    55:  'A-770041',
    56:  'WH-4-023',
    59:  'WZ-1-84',
    60:  'BI-2536',
    62:  'BMS-536924',
    63:  'BMS-509744',
    64:  'CMK',
    71:  'Pyrimethamine',
    83:  'JW-7-52-1',
    86:  'A-443654',
    87:  'GW843682X',
    88:  'Entinostat',
    89:  'Parthenolide',
    91:  'GSK319347A',
    94:  'TGX221',
    104: 'Bortezomib',
    106: 'XMD8-85',
    110: 'Seliciclib',
    111: 'Salubrinal',
    119: 'Lapatinib',
    127: 'GSK269962A',
    133: 'Doxorubicin',
    134: 'Etoposide',
    135: 'Gemcitabine',
    136: 'Mitomycin-C',
    140: 'Vinorelbine',
    147: 'NSC-87877',
    150: 'Bicalutamide',
    151: 'QS11',
    152: 'CP466722',
    153: 'Midostaurin',
    154: 'CHIR-99021',
    155: 'Ponatinib',
    156: 'AZD6482',
    157: 'JNK-9L',
    158: 'PF-562271',
    159: 'HG6-64-1',
    163: 'JQ1',
    164: 'JQ12',
    165: 'DMOG',
    166: 'FTI-277',
    167: 'OSU-03012',
    170: 'Shikonin',
    171: 'AKT inhibitor VIII',
    172: 'Embelin',
    173: 'FH535',
    175: 'PAC-1',
    176: 'IPA-3',
    177: 'GSK650394',
    178: 'BAY-61-3606',
    179: '5-Fluorouracil',
    180: 'Thapsigargin',
    182: 'Obatoclax Mesylate',
    184: 'BMS-754807',
    185: 'Linsitinib',
    186: 'Bexarotene',
    190: 'Bleomycin',
    192: 'LFM-A13',
    193: 'GW-2580',
    194: 'Luminespib',
    196: 'Phenformin',
    197: 'Bryostatin 1',
    199: 'Pazopanib',
    200: 'Dacinostat',
    201: 'Epothilone B',
    202: 'GSK1904529A',
    203: 'BMS-345541',
    204: 'Tipifarnib',
    205: 'Avagacestat',
    206: 'Ruxolitinib',
    207: 'AS601245',
    208: 'Ispinesib Mesylate',
    211: 'TL-2-105',
    219: 'AT-7519',
    221: 'TAK-715',
    222: 'BX-912',
    223: 'ZSTK474',
    224: 'AS605240',
    225: 'Genentech Cpd 10',
    226: 'GSK1070916',
    228: 'AKT inhibitor VIII',
    229: 'Enzastaurin',
    230: 'GSK429286A',
    231: 'FMK',
    235: 'QL-XII-47',
    238: 'Idelalisib',
    245: 'UNC0638',
    249: 'Cabozantinib',
    252: 'WZ3105',
    253: 'XMD14-99',
    254: 'Quizartinib',
    255: 'CP724714',
    256: 'JW-7-24-1',
    257: 'NPK76-II-72-1',
    258: 'STF-62247',
    260: 'NG-25',
    261: 'TL-1-85',
    262: 'VX-11e',
    263: 'FR-180204',
    265: 'Tubastatin A',
    266: 'Zibotentan',
    268: 'Sepantronium bromide',
    269: 'NSC-207895',
    271: 'VNLG/124',
    272: 'AR-42',
    273: 'CUDC-101',
    274: 'Belinostat',
    275: 'I-BET-762',
    276: 'CAY10603',
    277: 'Linifanib',
    279: 'BIX02189',
    281: 'Alectinib',
    282: 'Pelitinib',
    283: 'Omipalisib',
    286: 'KIN001-236',
    287: 'KIN001-244',
    288: 'WHI-P97',
    290: 'KIN001-260',
    291: 'KIN001-266',
    292: 'Masitinib',
    293: 'Amuvatinib',
    294: 'MPS-1-IN-1',
    295: 'NVP-BHG712',
    298: 'OSI-930',
    299: 'OSI-027',
    300: 'CX-5461',
    301: 'PHA-793887',
    302: 'PI-103',
    303: 'PIK-93',
    304: 'SB52334',
    305: 'TPCA-1',
    306: 'Fedratinib',
    308: 'WIKI4',
    309: 'Y-39983',
    310: 'YM201636',
    312: 'Tivozanib',
    326: 'GSK690693',
    328: 'SNX-2112',
    329: 'QL-XI-92',
    330: 'XMD13-2',
    331: 'QL-X-138',
    332: 'XMD15-27',
    333: 'T0901317',
    341: 'Selisistat',
    344: 'THZ-2-49',
    345: 'KIN001-270',
    346: 'THZ-2-102-1',
    1001:    'AICA Ribonucleotide',
    1003:    'Camptothecin',
    1004:    'Vinblastine',
    1005:    'Cisplatin',
    1006:    'Cytarabine',
    1007:    'Docetaxel',
    1008:    'Methotrexate',
    1009:    'Tretinoin',
    1010:    'Gefitinib',
    1011:    'Navitoclax',
    1012:    'Vorinostat',
    1013:    'Nilotinib',
    1014:    'Refametinib',
    1015:    'CI-1040',
    1016:    'Temsirolimus',
    1017:    'Olaparib',
    1018:    'Veliparib',
    1019:    'Bosutinib',
    1020:    'Lenalidomide',
    1021:    'Axitinib',
    1022:    'AZD7762',
    1023:    'GW441756',
    1024:    'Lestauritinib',
    1025:    'SB216763',
    1026:    'Tanespimycin',
    1028:    'VX-702',
    1029:    'Motesanib',
    1030:    'KU-55933',
    1031:    'Elesclomol',
    1032:    'Afatinib',
    1033:    'Vismodegib',
    1036:    'PLX-4720',
    1037:    'BX796',
    1038:    'NU7441',
    1039:    'SL0101',
    1042:    'Doramapimod',
    1043:    'JNK Inhibitor VIII',
    1046:    '681640',
    1047:    'Nutlin-3a (-)',
    1049:    'PD173074',
    1050:    'ZM447439',
    1052:    'RO-3306',
    1053:    'MK-2206',
    1054:    'Palbociclib',
    1057:    'Dactolisib',
    1058:    'Pictilisib',
    1059:    'AZD8055',
    1060:    'PD0325901',
    1061:    'SB590885',
    1062:    'Selumetinib',
    1066:    'AZD6482',
    1067:    'CCT007093',
    1069:    'EHT-1864',
    1072:    'Avagacestat',
    1091:    'BMS-536924',
    1114:    'Cetuximab',
    1129:    'PF-4708671',
    1133:    'Serdemetan',
    1142:    'HG-5-113-01',
    1143:    'HG-5-88-01',
    1149:    'TW 37',
    1158:    'XMD11-85h',
    1161:    'ZG-10',
    1164:    'XMD8-92',
    1166:    'QL-VIII-58',
    1170:    'CCT-018159',
    1175:    'Rucaparib',
    1192:    'GSK269962A',
    1194:    'SB-505124',
    1199:    'Tamoxifen',
    1203:    'QL-XII-61',
    1218:    'JQ1',
    1219:    'PFI-1',
    1230:    'IOX2',
    1236:    'UNC0638',
    1239:    'YK-4-279',
    1241:    'CHIR-99021',
    1242:    '(5Z)-7-Oxozeaenol',
    1243:    'Piperlongumine',
    1248:    'Daporinad',
    1259:    'Talazoparib',
    1261:    'rTRAIL',
    1262:    'UNC1215',
    1264:    'SGC0946',
    1268:    'XAV939',
    1371:    'PLX-4720',
    1372:    'Trametinib',
    1373:    'Dabrafenib',
    1375:    'Temozolomide',
    1377:    'Afatinib',
    1378:    'Bleomycin (50 uM)',
    1494:    'SN-38',
    1495:    'Olaparib',
    1498:    'Selumetinib',
    1502:    'Bicalutamide',
    1526:    'Refametinib',
    1527:    'Pictilisib',
    1529:    'Pevonedistat' }

GDSC_DRUG_TARGET_PATHWAYS = {
    1: 'EGFR signaling',
    3: 'PI3K/MTOR signaling',
    5: 'RTK signaling',
    6: 'RTK signaling',
    9: 'Protein stability and degradation',
    11: 'Mitosis',
    17: 'Other',
    29: 'ERK MAPK signaling',
    30: 'RTK signaling',
    32: 'Mitosis',
    34: 'RTK signaling',
    35: 'RTK signaling',
    37: 'RTK signaling',
    38: 'RTK signaling',
    41: 'Mitosis',
    45: 'Other',
    51: 'Other',
    52: 'ABL signaling',
    53: 'Cell cycle',
    54: 'Cell cycle',
    55: 'Other, kinases',
    56: 'Other, kinases',
    59: 'Other, kinases',
    60: 'Cell cycle',
    62: 'IGFR signaling',
    63: 'Other',
    64: 'ERK MAPK signaling',
    71: 'Other',
    83: 'PI3K/MTOR signaling',
    86: 'PI3K/MTOR signaling',
    87: 'Cell cycle',
    88: 'Chromatin histone acetylation',
    89: 'Chromatin histone acetylation',
    91: 'Other',
    94: 'PI3K/MTOR signaling',
    104: 'Protein stability and degradation',
    106: 'Other',
    110: 'Cell cycle',
    111: 'Other',
    119: 'EGFR signaling',
    127: 'Cytoskeleton',
    133: 'DNA replication',
    134: 'DNA replication',
    135: 'DNA replication',
    136: 'DNA replication',
    140: 'Mitosis',
    147: 'Other',
    150: 'Hormone-related',
    151: 'Other',
    152: 'Genome integrity',
    153: 'Other',
    154: 'WNT signaling',
    155: 'RTK signaling',
    156: 'PI3K/MTOR signaling',
    157: 'JNK and p38 signaling',
    158: 'Cytoskeleton',
    159: 'ERK MAPK signaling',
    163: 'Chromatin other',
    164: 'Chromatin histone acetylation',
    165: 'Metabolism',
    166: 'Other',
    167: 'Other, kinases',
    170: 'Other',
    171: 'PI3K/MTOR signaling',
    172: 'Apoptosis regulation',
    173: 'WNT signaling',
    175: 'Apoptosis regulation',
    176: 'Cytoskeleton',
    177: 'Other',
    178: 'Other, kinases',
    179: 'Other',
    180: 'Other',
    182: 'Apoptosis regulation',
    184: 'IGFR signaling',
    185: 'IGFR signaling',
    186: 'Other',
    190: 'DNA replication',
    192: 'Other, kinases',
    193: 'RTK signaling',
    194: 'Protein stability and degradation',
    196: 'Other',
    197: 'Other, kinases',
    199: 'RTK signaling',
    200: 'Chromatin histone acetylation',
    201: 'Mitosis',
    202: 'IGFR signaling',
    203: 'Other, kinases',
    204: 'Other',
    205: 'Other',
    206: 'Other, kinases',
    207: 'JNK and p38 signaling',
    208: 'Mitosis',
    211: 'ERK MAPK signaling',
    219: 'Cell cycle',
    221: 'JNK and p38 signaling',
    222: 'PI3K/MTOR signaling',
    223: 'PI3K/MTOR signaling',
    224: 'PI3K/MTOR signaling',
    225: 'Mitosis',
    226: 'Mitosis',
    228: 'PI3K/MTOR signaling',
    229: 'Other, kinases',
    230: 'Cytoskeleton',
    231: 'Other, kinases',
    235: 'Other, kinases',
    238: 'PI3K/MTOR signaling',
    245: 'Chromatin histone methylation',
    249: 'Other, kinases',
    252: 'Other',
    253: 'Other',
    254: 'RTK signaling',
    255: 'EGFR signaling',
    256: 'Other, kinases',
    257: 'Cell cycle',
    258: 'Other',
    260: 'Other, kinases',
    261: 'Other, kinases',
    262: 'ERK MAPK signaling',
    263: 'ERK MAPK signaling',
    265: 'Chromatin histone acetylation',
    266: 'Other',
    268: 'Apoptosis regulation',
    269: 'p53 pathway',
    271: 'Chromatin histone acetylation',
    272: 'Chromatin histone acetylation',
    273: 'Other',
    274: 'Chromatin histone acetylation',
    275: 'Chromatin other',
    276: 'Chromatin histone acetylation',
    277: 'RTK signaling',
    279: 'ERK MAPK signaling',
    281: 'RTK signaling',
    282: 'EGFR signaling',
    283: 'PI3K/MTOR signaling',
    286: 'Other',
    287: 'Other, kinases',
    288: 'Other, kinases',
    290: 'Other',
    291: 'Other, kinases',
    292: 'Other, kinases',
    293: 'Other, kinases',
    294: 'Mitosis',
    295: 'Other',
    298: 'RTK signaling',
    299: 'PI3K/MTOR signaling',
    300: 'Other',
    301: 'Cell cycle',
    302: 'PI3K/MTOR signaling',
    303: 'PI3K/MTOR signaling',
    304: 'RTK signaling',
    305: 'Other, kinases',
    306: 'Other, kinases',
    308: 'WNT signaling',
    309: 'Cytoskeleton',
    310: 'Other',
    312: 'RTK signaling',
    326: 'PI3K/MTOR signaling',
    328: 'Protein stability and degradation',
    329: 'Other',
    330: 'Apoptosis regulation',
    331: 'Other, kinases',
    332: 'Other, kinases',
    333: 'Other',
    341: 'Chromatin histone acetylation',
    344: 'Cell cycle',
    345: 'Cell cycle',
    346: 'Cell cycle',
    1001: 'Metabolism',
    1003: 'DNA replication',
    1004: 'Mitosis',
    1005: 'DNA replication',
    1006: 'DNA replication',
    1007: 'Mitosis',
    1008: 'DNA replication',
    1009: 'Other',
    1010: 'EGFR signaling',
    1011: 'Apoptosis regulation',
    1012: 'Chromatin histone acetylation',
    1013: 'ABL signaling',
    1014: 'ERK MAPK signaling',
    1015: 'ERK MAPK signaling',
    1016: 'PI3K/MTOR signaling',
    1017: 'Genome integrity',
    1018: 'Genome integrity',
    1019: 'Other, kinases',
    1020: 'Protein stability and degradation',
    1021: 'RTK signaling',
    1022: 'Cell cycle',
    1023: 'RTK signaling',
    1024: 'Other, kinases',
    1025: 'WNT signaling',
    1026: 'Protein stability and degradation',
    1028: 'JNK and p38 signaling',
    1029: 'RTK signaling',
    1030: 'Genome integrity',
    1031: 'Protein stability and degradation',
    1032: 'EGFR signaling',
    1033: 'Other',
    1036: 'ERK MAPK signaling',
    1037: 'Other',
    1038: 'Genome integrity',
    1039: 'Other',
    1042: 'JNK and p38 signaling',
    1043: 'JNK and p38 signaling',
    1046: 'Cell cycle',
    1047: 'p53 pathway',
    1049: 'RTK signaling',
    1050: 'Mitosis',
    1052: 'Cell cycle',
    1053: 'PI3K/MTOR signaling',
    1054: 'Cell cycle',
    1057: 'PI3K/MTOR signaling',
    1058: 'PI3K/MTOR signaling',
    1059: 'PI3K/MTOR signaling',
    1060: 'ERK MAPK signaling',
    1061: 'ERK MAPK signaling',
    1062: 'ERK MAPK signaling',
    1066: 'PI3K/MTOR signaling',
    1067: 'Other',
    1069: 'Cytoskeleton',
    1072: 'Other',
    1091: 'IGFR signaling',
    1114: 'EGFR signaling',
    1129: 'PI3K/MTOR signaling',
    1133: 'p53 pathway',
    1142: 'Other',
    1143: 'Other, kinases',
    1149: 'Apoptosis regulation',
    1158: 'Other',
    1161: 'JNK and p38 signaling',
    1164: 'Other, kinases',
    1166: 'Other',
    1170: 'Protein stability and degradation',
    1175: 'Genome integrity',
    1192: 'Cytoskeleton',
    1194: 'RTK signaling',
    1199: 'Hormone-related',
    1203: 'Other, kinases',
    1218: 'Chromatin other',
    1219: 'Chromatin other',
    1230: 'Other',
    1236: 'Chromatin histone methylation',
    1239: 'Other',
    1241: 'WNT signaling',
    1242: 'Other, kinases',
    1243: 'Other',
    1248: 'Metabolism',
    1259: 'Genome integrity',
    1261: 'Apoptosis regulation',
    1262: 'Chromatin other',
    1264: 'Chromatin histone methylation',
    1268: 'WNT signaling',
    1371: 'ERK MAPK signaling',
    1372: 'ERK MAPK signaling',
    1373: 'ERK MAPK signaling',
    1375: 'DNA replication',
    1377: 'EGFR signaling',
    1378: 'DNA replication',
    1494: 'DNA replication',
    1495: 'Genome integrity',
    1498: 'ERK MAPK signaling',
    1502: 'Hormone-related',
    1526: 'ERK MAPK signaling',
    1527: 'PI3K/MTOR signaling',
    1529: 'Other'}


