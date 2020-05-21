'''
More factorization code, courtesy of Jackson Loper.
'''
import numpy as np
import dataclasses

def moments(muhat_row,Sighat_row,muhat_col,Sighat_col,**kwargs):
    row_m2 = Sighat_row + np.einsum('ij,ik->ijk',muhat_row,muhat_row)
    col_m2 = Sighat_col + np.einsum('ij,ik->ijk',muhat_col,muhat_col)

    mn= muhat_row @ muhat_col.T
    e2 = np.einsum('ajk,bjk -> ab',row_m2,col_m2)
    vr = e2-mn**2

    return row_m2,col_m2,mn,vr

def prior_KL(muhat,Sighat,mu,Sig):
    df = muhat - mu[None,:]
    m2 = Sighat + np.einsum('ij,ik->ijk',df,df)

    mahaltr = np.sum(np.linalg.inv(Sig)[None,:,:]*m2)

    nobs = np.prod(muhat.shape)

    sigdet_prior = muhat.shape[0]*np.linalg.slogdet(Sig)[1]
    sigdet_post = np.sum(np.linalg.slogdet(Sighat)[1])

    return .5 * (mahaltr - nobs + sigdet_prior - sigdet_post)


r'''
     _ _                 _       _     
  __| (_)___ _ __   __ _| |_ ___| |__  
 / _` | / __| '_ \ / _` | __/ __| '_ \ 
| (_| | \__ \ |_) | (_| | || (__| | | |
 \__,_|_|___/ .__/ \__,_|\__\___|_| |_|
            |_|                        
'''

def ELBO_dataterm(X,mn,vr,kind,theta=None):
    if kind=='normal':
        return loss_normal(X,mn,vr,theta)
    elif kind=='bernoulli':
        return loss_bernoulli(X,mn,vr)
    else:
        raise Exception("NYI")

def accumulate_omega_for_rows(X,muhat_row,Sighat_row,muhat_col,Sighat_col,kind,theta=None):
    row_m2,col_m2,mn,vr= moments(muhat_row,Sighat_row,muhat_col,Sighat_col)
    xi1,xi2=get_xi(X,mn,vr,kind,theta) # <-- Nrow x Ncol
    omega_2 = np.einsum('rc,cij -> rij',xi2,col_m2)
    omega_1 = np.einsum('rc,ci -> ri',xi1,muhat_col)

    return omega_1,omega_2

def accumulate_omega_for_cols(X,muhat_row,Sighat_row,muhat_col,Sighat_col,kind,theta=None):
    row_m2,col_m2,mn,vr= moments(muhat_row,Sighat_row,muhat_col,Sighat_col)
    xi1,xi2=get_xi(X,mn,vr,kind,theta) # <-- Nrow x Ncol
    omega_2 = np.einsum('rc,rij -> cij',xi2,row_m2)
    omega_1 = np.einsum('rc,ri -> ci',xi1,muhat_row)
    return omega_1,omega_2

def get_xi(X,mn,vr,kind,theta=None):
    if kind=='normal':
        return get_xi_normal(X,mn,vr,theta)
    elif kind=='bernoulli':
        return get_xi_bernoulli(X,mn,vr)
    else:
        raise Exception("NYI")

def get_new_theta(X,mn,vr,kind,theta=None):
    if kind=='normal':
        zeta = get_zeta_normal(X,mn,vr)
        return np.sqrt(np.mean(zeta,axis=0))
    elif kind=='bernoulli':
        return None
    else:
        raise Exception("NYI")

r'''
     _       _        _                          
  __| | __ _| |_ __ _| |_ ___ _ __ _ __ ___  ___ 
 / _` |/ _` | __/ _` | __/ _ \ '__| '_ ` _ \/ __|
| (_| | (_| | || (_| | ||  __/ |  | | | | | \__ \
 \__,_|\__,_|\__\__,_|\__\___|_|  |_| |_| |_|___/
                                                 
'''

def loss_normal(X,curmu,curvar,theta):
    zeta = (X-curmu)**2 + curvar
    return -.5*zeta/(2*theta**2) -.5*np.log(np.pi*2*theta**2)

def loss_bernoulli(X,curmu,curvar):
    return (X-.5)*curmu - log2cosho2_safe(np.sqrt(curmu**2+curvar**2))

def get_zeta_normal(X,curmu,curvar):
    return (X-curmu)**2 + curvar

def solve_zeta_normal(zeta):
    return np.mean(zeta,axis=0)

def get_xi_normal(X,curmu,curvar,theta):
    '''
    Input
    - X      Nrow x Ncol
    - curmu  Nrow x Ncol
    - curvar Nrow x Ncol
    - theta  Nrow x Ncol  (or broadcastable to)

    Output
    - xi1    Nrow x Ncol
    - xi2    Nrow x Ncol
    '''

    return X/theta**2,np.outer(np.ones(X.shape[0]),1/theta**2)

def get_xi_bernoulli(X,curmu,curvar):
    '''
    Input
    - X      Nrow x Ncol
    - curmu  Nrow x Ncol
    - curvar Nrow x Ncol

    Output
    - xi1
    - xi2
    '''

    gamsq = curmu**2 + curvar
    gam = np.sqrt(gamsq)

    xi2 = pge_safe(gam)
    xi1 = (X-.5)

    return xi1,xi2

def pge_safe(x):
    switch=np.abs(x)<.00001

    A=.25-0.020833333333333332*(x**2)
    B=np.tanh(x/2)/(2*x)
    return np.where(switch,A,B)

def log2cosho2_safe(x):
    '''
    returns log(2*(cosh(x/2))
    '''

    return np.log(2*np.cosh(x/2))