"""Nested response-one policies; all dates refer to realized monthly returns.
The baseline is profiled out exactly. Complexity is managed-payoff EDF only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh

BUDGETS = np.array([0., .5, 1., 2., 4., 8., 16., 32.])

@dataclass
class Fit:
    weights: np.ndarray
    unhedged_weights: np.ndarray
    baseline: np.ndarray
    complexity: np.ndarray
    ridge: np.ndarray
    residual_second_moment: np.ndarray
    hedge: np.ndarray


def fit_extension(returns: np.ndarray, n_base: int, budgets: np.ndarray = BUDGETS) -> Fit:
    """Return one native response-one policy per incremental complexity budget.

    returns[:, :n_base] are protected baseline directions. Excluded directions
    have zero coefficients, so c=0 exactly nests the estimated baseline.
    This routine does not see the observation on which weights will be evaluated.
    """
    R=np.asarray(returns,dtype=float); c=np.asarray(budgets,dtype=float)
    if R.ndim!=2 or len(R)<=n_base or not np.isfinite(R).all():
        raise ValueError('Need finite training returns and more dates than baseline columns')
    if not 0 < n_base <= R.shape[1] or c.ndim!=1 or np.any(c<0):
        raise ValueError('Invalid baseline dimension or complexity grid')
    B=R[:,:n_base]; X=R[:,n_base:]; T=len(R)
    if np.linalg.matrix_rank(B)!=n_base:raise ValueError('Baseline is rank deficient')
    a0=np.linalg.lstsq(B,np.ones(T),rcond=None)[0]
    hedge=np.linalg.lstsq(B,X,rcond=None)[0]
    H=X-B@hedge
    S=H.T@H/T; mu=H.mean(0)
    q=len(X.T); w=np.zeros((R.shape[1],len(c)));w[:n_base]=a0[:,None]
    unhedged=w.copy(); ridge=np.full(len(c),np.inf); actual=np.zeros(len(c))
    if q:
        eig,U=eigh(S,check_finite=False); top=float(max(eig[-1],0))
        eig=np.where(eig>max(top,1e-30)*1e-11,eig,0); rank=int(np.count_nonzero(eig))
        if rank:
            use=np.flatnonzero(c>0); wanted=np.minimum(c[use],rank*.98)
            low=np.full(len(use),np.log(top)-35.);high=np.full(len(use),np.log(top)+35.)
            for _ in range(55):
                mid=(low+high)/2;cc=(eig[:,None]/(eig[:,None]+np.exp(mid))).sum(0)
                low=np.where(cc>wanted,mid,low);high=np.where(cc>wanted,high,mid)
            lam=np.exp((low+high)/2); b=U@((U.T@mu)[:,None]/(eig[:,None]+lam))
            w[n_base:,use]=b;w[:n_base,use]-=hedge@b
            unhedged[n_base:,use]=b
            ridge[use]=lam;actual[use]=wanted
    return Fit(w,unhedged,a0,n_base+actual,ridge,S,hedge)


def population_metrics(w: np.ndarray, mean: np.ndarray, second: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Exact moment evaluation: Q and annualized SR."""
    if w.ndim==1:w=w[:,None]
    m=mean@w;s=np.einsum('il,ij,jl->l',w,second,w)
    q=1-2*m+s;v=np.maximum(s-m*m,1e-20)
    return q,np.sqrt(12)*m/np.sqrt(v)


def sharpe(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    return np.sqrt(12)*x.mean(0)/np.maximum(x.std(0,ddof=1),1e-14)


def circular_indices(n: int,reps: int,block: int,seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]


def admit_extension(gains: np.ndarray, *, seed: int=1903, repetitions: int=400, block: int=12) -> tuple[int,np.ndarray]:
    """Simultaneous one-sided lower bounds for held-out mean loss improvements.

    Columns are locked challengers; larger is better. -1 means none clears zero.
    This is an experimental admission rule, not a finite-sample coverage theorem.
    """
    gains=np.asarray(gains,float)
    if gains.ndim!=2 or len(gains)<24 or not np.isfinite(gains).all():
        raise ValueError('Admission requires finite, paired confirmation data')
    observed=gains.mean(0)
    idx=circular_indices(len(gains),repetitions,block,seed)
    means=gains[idx].mean(1)
    radius=np.quantile(np.max(means-observed,axis=1),.95)
    lower=observed-radius
    chosen=int(np.argmax(lower)) if np.max(lower)>0 else -1
    return chosen,lower


def paired_inference(base: np.ndarray, extended: np.ndarray, *, seed: int=528, repetitions: int=2000,block: int=12) -> dict:
    a=np.asarray(base,float);b=np.asarray(extended,float)
    if b.ndim==1:b=b[:,None]
    if len(a)!=len(b):raise ValueError('Nonpaired OOS samples')
    idx=circular_indices(len(a),repetitions,block,seed)
    draws=sharpe_batch(b[idx])-sharpe_batch(a[idx,None])
    delta=sharpe(b)-sharpe(a)
    loss_gain=(1-a[:,None])**2-(1-b)**2
    loss_draws=loss_gain[idx].mean(1);lg=loss_gain.mean(0)
    dlo,dhi=np.quantile(draws,[.025,.975],axis=0)
    qlo,qhi=np.quantile(loss_draws,[.025,.975],axis=0)
    radius=np.quantile(np.max(np.abs(draws-delta),axis=1),.95)
    return dict(delta_sr=delta,low_sr=dlo,high_sr=dhi,sim_low_sr=delta-radius,sim_high_sr=delta+radius,
                gain_loss=lg,low_loss=qlo,high_loss=qhi)


def sharpe_batch(x: np.ndarray)->np.ndarray:
    return np.sqrt(12)*x.mean(1)/np.maximum(x.std(1,ddof=1),1e-14)
