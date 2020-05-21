'''
Factorization code courtesy of Jackson Loper
'''
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
import lowlevel
import scipy.sparse.linalg
import numpy.random as npr
import dataclasses
import scipy as sp
import time
import threading
import traceback

class Trainer:
    def __init__(self,mod):
        self.mod=mod
        self.elbos=[mod.ELBO()]
        self.actions=['init']
        self.elbo_action_crossreference=[0]
        self._times=[time.time()]
        self._th=None
        self._err=None
        self._keepgoing=False

    @property
    def ELBO_times(self):
        return np.array(self._times)[self.elbo_action_crossreference] - self._times[0]

    def collect_ELBO(self):
        self.elbos.append(self.mod.ELBO())
        self.elbo_action_crossreference.append(len(self.actions)-1)

    def update(self,nm,ELBO=False):
        getattr(self.mod,'update_'+nm)()
        self.actions.append(nm)
        self._times.append(time.time())
        if ELBO:
            self.collect_ELBO()

    update_types=['rows','cols','prior_rows','prior_cols','thetas']

    def sweep(self):
        for nm in self.update_types:
            self.update(nm)

    def thread_status(self):
        print("alive?",self._th.is_alive(),f'    nactions={len(self.actions)}')
        if self._err is not None:
            print(self._err)
    def stop_thread(self,complain=False):
        if (self._th is None) or (not self._th.is_alive()):
            if complain:
                raise Exception("nothing running")
        else:
            self._keepgoing=False
            self._th.join()
    def train_thread(self):
        if (self._th is not None) and (self._th.is_alive()):
            raise Exception("already running")
        self._keepgoing=True
        self._err=None
        def go():
            while True:
                try:
                    for nm in self.update_types:
                        self.update(nm)
                        nats= self.elbos[-1]['nats']
                        if not self._keepgoing:
                            return 
                    self.collect_ELBO()
                except Exception as e:
                    self._keepgoing=False
                    self._err=(e,traceback.format_exc())
                    raise
        self._th=threading.Thread(target=go)
        self._th.start()

    @property
    def nats(self):
        return np.array([x['nats'] for x in self.elbos])
    

@dataclasses.dataclass
class Block:
    kind: str
    data: np.ndarray
        
@dataclasses.dataclass
class BlockData:
    blocks: list
    nrows: int
    ncols: int
        
    @property
    def nobs(self):
        return self.nrows*self.ncols
        
    @property
    def colbins(self):
        bins=[]
        i=0
        for b in self.blocks:
            j=i+b.data.shape[1]
            bins.append([i,j])
            i=j
        return np.array(bins)
        
@dataclasses.dataclass
class PosteriorGaussian:
    muhat: np.ndarray
    Sighat: np.ndarray
    muhat_velocity: np.ndarray = 0.0
    muhat_momentum_factor: np.ndarray = 0.0

    def update_muhat(self,newmuhat):
        delta = newmuhat - self.muhat
        self.muhat_velocity = delta + self.muhat_velocity*self.muhat_momentum_factor
        self.muhat = self.muhat + self.muhat_velocity

    def snapshot(self):
        return self.muhat.copy(),self.Sighat.copy()
        
@dataclasses.dataclass
class PriorGaussian:
    mu: np.ndarray
    Sig: np.ndarray

    def snapshot(self):
        return self.mu.copy(),self.Sig.copy()
        
@dataclasses.dataclass
class GaussianMatrixVI:
    post: PosteriorGaussian
    prior: PriorGaussian
        
    def update_prior(self):
        self.prior.mu=np.mean(self.post.muhat,axis=0)
        self.prior.Sig=np.cov(self.post.muhat.T,ddof=0) + np.mean(self.post.Sighat,axis=0)
        self.prior.Sig = np.eye(self.prior.Sig.shape[0])*(1e-8) + (1-1e-8)*self.prior.Sig
    
    def kl(self):
        return lowlevel.prior_KL(self.post.muhat,self.post.Sighat,self.prior.mu,self.prior.Sig)

    def snapshot(self):
        return dict(post=self.post.snapshot(),prior=self.prior.snapshot())

    @classmethod
    def load(cls,snap):
        return GaussianMatrixVI(
            PosteriorGaussian(snap['post'][0].copy(),snap['post'][1].copy()),
            PriorGaussian(snap['prior'][0].copy(),snap['prior'][1].copy())
        )

@dataclasses.dataclass
class Model:
    data: BlockData
    rowinfo: GaussianMatrixVI
    colinfo: list # <-- one GaussianMatrixVI for each block
    thetas: list # <-- one ndarray for each block

    def snapshot(self):
        return dict(
            rowinfo=self.rowinfo.snapshot(),
            colinfo=[x.snapshot() for x in self.colinfo],
            thetas=[(x.copy() if (x is not None) else None) for x in self.thetas]
        )
        
    @classmethod
    def load(cls,data,snap):
        return Model(
            data,
            GaussianMatrixVI.load(snap['rowinfo']),
            [GaussianMatrixVI.load(x) for x in snap['colinfo']],
            [(x.copy() if (x is not None) else None) for x in snap['thetas']]
        )

    @property
    def row_loading(self):
        return self.rowinfo.post.muhat
    @property
    def col_loading(self):
        return np.concatenate([x.post.muhat for x in self.colinfo],axis=0)

    def msms(self,ell):
        st,en = self.data.colbins[ell]
        return dict(
            muhat_row=self.rowinfo.post.muhat,
            Sighat_row=self.rowinfo.post.Sighat,
            muhat_col=self.colinfo[ell].post.muhat,
            Sighat_col=self.colinfo[ell].post.Sighat,
            kind=self.data.blocks[ell].kind,
            theta=self.thetas[ell]
        )
        
    def update_rows(self):
        omega1=np.linalg.solve(self.rowinfo.prior.Sig,self.rowinfo.prior.mu)[None,:]
        omega2=np.linalg.inv(self.rowinfo.prior.Sig)[None,:,:]
        for i,(th,block) in enumerate(zip(self.thetas,self.data.blocks)):
            o1,o2=lowlevel.accumulate_omega_for_rows(block.data,**self.msms(i))
            omega1=omega1 + o1
            omega2=omega2 + o2

        self.rowinfo.post.update_muhat(np.linalg.solve(omega2,omega1))
        self.rowinfo.post.Sighat = np.linalg.inv(omega2)
       
    def update_prior_rows(self):
        self.rowinfo.update_prior()

    def update_prior_cols(self):
        for c in self.colinfo:
            c.update_prior()
    
    def update_cols(self):
        for i,(th,block,bn) in enumerate(zip(self.thetas,self.data.blocks,self.data.colbins)):
            omega1=np.linalg.solve(self.colinfo[i].prior.Sig,self.colinfo[i].prior.mu)
            omega2=np.linalg.inv(self.colinfo[i].prior.Sig)
            o1,o2=lowlevel.accumulate_omega_for_cols(block.data,**self.msms(i))

            self.colinfo[i].post.update_muhat(np.linalg.solve(omega2+o2,omega1+o1))
            self.colinfo[i].post.Sighat = np.linalg.inv(omega2+o2)
            
    def update_thetas(self):
        for i,(th,block) in enumerate(zip(self.thetas,self.data.blocks)):
            if th is not None:
                row_m2,col_m2,mn,vr=lowlevel.moments(**self.msms(i))
                th[:]=lowlevel.get_new_theta(block.data,mn,vr,block.kind,th)
            
    def ELBO(self):
        dataterm = 0
        for i,(th,block) in enumerate(zip(self.thetas,self.data.blocks)):
            row_m2,col_m2,mn,vr=lowlevel.moments(**self.msms(i))
            dataterm+=np.sum(lowlevel.ELBO_dataterm(block.data,mn,vr,block.kind,th))
            
        kl_row = self.rowinfo.kl()
        kl_cols = np.array([x.kl() for x in self.colinfo])
        
        nats = -(dataterm -kl_row - np.sum(kl_cols)) / (self.data.nobs)
        
        return dict(nats=nats,kl_row=kl_row,kl_cols=kl_cols,dataterm=dataterm)
        
def initialize_half(U):
    Sig = np.diag(np.var(U,axis=0))
    
    return GaussianMatrixVI(
        post = PosteriorGaussian(
            muhat = U,
            Sighat = ((.1)**2)*np.tile(Sig,[U.shape[0],1,1])*np.std(U)
        ),
        prior = PriorGaussian(
            mu = np.mean(U,axis=0),
            Sig = np.cov(U.T),
        )
    )

def initialize(data,Nk):
    tots=[]
    thetas=[]
    for block in data.blocks:
        if block.kind=='normal':
            tots.append(block.data)
            thetas.append(np.var(block.data,axis=0))
        elif block.kind=='bernoulli':
            tots.append(2*(block.data-.5))
            thetas.append(None)
        else:
            raise Exception("NYI")
    
    tot = np.concatenate(tots,axis=1)
    U,e,V=sp.sparse.linalg.svds(tot,Nk)
    V=V.T
    U=U@np.diag(np.sqrt(e))
    V=V@np.diag(np.sqrt(e))
    
    return Model(
        data,
        initialize_half(U),
        [initialize_half(V[st:en]) for (st,en) in data.colbins],
        thetas=thetas
    )


