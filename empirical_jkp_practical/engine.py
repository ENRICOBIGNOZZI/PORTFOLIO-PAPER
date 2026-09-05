"""Causal response-one experiments on public characteristic portfolios.
C is managed-payoff effective dimension, not feature count. No stock-level claim.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist

CALIBRATION_END='1994-12-31'
VALIDATION_START='1995-01-01'
VALIDATION_END='2004-12-31'
TEST_START='2005-01-01'
DATA_START='1973-01-01'
TARGET_VOL=.10
SLEEVE_GROSS_CAP=3.
COV_SHRINK=.10
BASE_BUDGETS=np.array([.5,1,2,4,8,12,16,24,32,48,64,96,118.])

@dataclass
class StudyData:
    returns: pd.DataFrame
    state: pd.DataFrame
    metadata: pd.DataFrame
    calibration_mean: np.ndarray
    calibration_sd: np.ndarray
    ell: float
    universe_audit: pd.DataFrame

def sr(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    return np.sqrt(12.)*x.mean(axis=0)/np.maximum(x.std(axis=0,ddof=1),1e-12)

def load_public(raw: Path,weighting: str='vw_cap') -> tuple[pd.DataFrame,pd.DataFrame]:
    files=list((raw/f'usa_all_factors_{weighting}').glob('*.csv'))
    if len(files)!=1:raise ValueError('Expected one official USA factor CSV')
    d=pd.read_csv(files[0],parse_dates=['date'])
    if d.duplicated(['date','name']).any():raise ValueError('Duplicate factor-date records')
    # Official ret already includes the original-paper long-short orientation.
    panel=d.pivot(index='date',columns='name',values='ret').sort_index().loc[DATA_START:]
    panel.index=panel.index.to_period('M').to_timestamp('M')
    expected=pd.period_range(panel.index.min(),panel.index.max(),freq='M').to_timestamp('M')
    panel=panel.reindex(expected)
    metadata=(pd.read_csv(raw/'factor_details.csv').dropna(subset=['abr_jkp']).drop_duplicates('abr_jkp').set_index('abr_jkp'))
    if not panel.columns.isin(metadata.index).all():raise ValueError('Unknown characteristic name')
    metadata=metadata.reindex(panel.columns).copy()
    metadata['group']=metadata['group'].str.lower().str.replace(' ','_',regex=False)
    def pubyear(s):
        years=re.findall(r'\b(?:19|20)\d{2}\b',str(s))
        return int(years[0]) if years else np.nan
    metadata['publication_year']=metadata['cite'].map(pubyear)
    return panel,metadata

def make_state(panel: pd.DataFrame,metadata: pd.DataFrame) -> pd.DataFrame:
    groups={g:panel.loc[:,metadata.index[metadata['group']==g]].mean(axis=1) for g in sorted(metadata['group'].unique())}
    themes=pd.DataFrame(groups,index=panel.index)
    # Row dated t is selected BEFORE the return in month t is observed.
    return pd.concat([themes.rolling(12,min_periods=12).mean().shift(1).add_suffix('__mean12'),themes.rolling(12,min_periods=12).std(ddof=1).shift(1).add_suffix('__vol12')],axis=1)

def prepare(raw: Path,weighting: str='vw_cap',universe: list[str]|None=None) -> StudyData:
    panel,meta=load_public(raw,weighting)
    cal=panel.loc[:CALIBRATION_END]
    eligible=cal.notna().all()&(cal.std()>1e-8)
    audit=pd.DataFrame({'characteristic':panel.columns,'calibration_complete':cal.notna().all().values,'calibration_nonconstant':(cal.std()>1e-8).values,'eligible':eligible.values})
    names=list(panel.columns[eligible]) if universe is None else universe
    if not names or not set(names).issubset(panel.columns):raise ValueError('Invalid calibration universe')
    panel=panel.loc[:,names];meta=meta.loc[names]
    if panel.isna().any().any():raise ValueError('Missing return in preselected universe: explicit policy required')
    state=make_state(panel,meta).dropna();panel=panel.loc[state.index]
    xcal=state.loc[:CALIBRATION_END].to_numpy(float)
    mu=xcal.mean(0);sd=np.maximum(xcal.std(0),1e-8)
    ell=float(np.median(pdist((xcal-mu)/sd)))
    if not np.isfinite(ell) or ell<=0:raise ValueError('Degenerate state calibration')
    return StudyData(panel,state,meta,mu,sd,ell,audit)

def kernel(data: StudyData,kind: str='matern',state_columns: list[str]|None=None,rff_p: int|None=None,seed: int=739) -> np.ndarray:
    mask=np.ones(data.state.shape[1],dtype=bool) if state_columns is None else data.state.columns.isin(state_columns)
    x=(data.state.to_numpy(float)[:,mask]-data.calibration_mean[mask])/data.calibration_sd[mask]
    if kind=='constant':return np.ones((len(x),len(x)))
    if x.shape[1]==0:raise ValueError('No state coordinates')
    # Length scale and preprocessing frozen at pre-validation values.
    if rff_p is not None:
        if rff_p%2 or rff_p<2 or rff_p>1024:raise ValueError('Paired RFF P must be even and <=1024')
        rng=np.random.default_rng(seed);w=rng.normal(size=(512,data.state.shape[1]))
        w=w/np.sqrt(rng.chisquare(3,size=512)/3)[:,None]/data.ell
        z=x@w[:rff_p//2,mask].T
        phi=np.concatenate([np.cos(z),np.sin(z)],axis=1)/np.sqrt(rff_p/2)
        return phi@phi.T
    if kind=='linear':return 1+x@x.T/data.state.shape[1]
    distance=cdist(x,x)/data.ell
    if kind=='rbf':return np.exp(-.5*distance**2)
    if kind=='matern':
        u=np.sqrt(3.)*distance
        return (1+u)*np.exp(-u)
    raise ValueError(kind)

def lambdas_for_complexity(eig: np.ndarray,targets: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    eig=np.maximum(np.asarray(eig,float),0);top=float(eig.max())
    if top<=0:raise ValueError('Zero managed-payoff operator')
    rank=np.count_nonzero(eig>top*1e-12);eig=np.where(eig>top*1e-12,eig,0)
    actual=np.minimum(np.asarray(targets,float),rank*(1-1e-7))
    if np.any(actual<=0):raise ValueError('Complexity must be positive')
    lo=np.full(len(actual),np.log(top)-35);hi=np.full(len(actual),np.log(top)+35)
    for _ in range(50):
        mid=(lo+hi)/2;c=(eig[:,None]/(eig[:,None]+np.exp(mid)[None,:])).sum(0)
        lo=np.where(c>actual,mid,lo);hi=np.where(c>actual,hi,mid)
    return np.exp((lo+hi)/2),actual

def scale_weights(raw: np.ndarray,history: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    centered=history-history.mean(0);sample=centered@raw
    variance=(1-COV_SHRINK)*(sample**2).sum(0)/(len(history)-1)
    variance+=COV_SHRINK*(history.var(0,ddof=1)[:,None]*raw**2).sum(0)
    variance+=1e-10*(raw**2).sum(0)
    multiplier=TARGET_VOL/np.sqrt(np.maximum(12*variance,1e-20))
    multiplier=np.minimum(multiplier,SLEEVE_GROSS_CAP/np.maximum(np.abs(raw).sum(0),1e-20))
    return raw*multiplier,multiplier

def rolling(data: StudyData,K: np.ndarray,*,window: int=120,assets: list[str]|None=None,budgets: np.ndarray|None=None,end: str|None=None,diagnostics: bool=False) -> dict:
    columns=list(data.returns.columns) if assets is None else assets
    R=data.returns.loc[:,columns].to_numpy(float);dates=data.returns.index
    b=BASE_BUDGETS if budgets is None else np.asarray(budgets,float)
    b=np.unique(np.minimum(b,window*.985))
    ii=np.flatnonzero((dates>=VALIDATION_START)&(np.arange(len(dates))>=window))
    if end is not None:ii=ii[dates[ii]<=end]
    if len(ii)==0:raise ValueError('No evaluation history')
    shape=(len(ii),len(b))
    arrays={k:np.empty(shape) for k in ['returns','native_loss','turnover_proxy','complexity','lambda','is_sharpe','gross']}
    previous=np.zeros((len(columns),len(b)))
    if diagnostics:
        arrays['spectral_share']=np.zeros((len(ii),window));arrays['contributions']=np.zeros((len(ii),len(b),4))
        arrays['weights']=np.zeros((len(ii),len(columns),len(b)),dtype=np.float32)
    for row,t in enumerate(ii):
        history=R[t-window:t]
        G=K[t-window:t,t-window:t]*(history@history.T)/window
        eig,U=eigh(G,check_finite=False,driver='evr')
        eig=np.where(eig>max(eig[-1],1e-30)*1e-12,eig,0)
        lam,actual=lambdas_for_complexity(eig,b);u1=U.sum(0)
        alpha=(U@(u1[:,None]/(eig[:,None]+lam[None,:])))/window
        weights_raw=history.T@(K[t,t-window:t,None]*alpha)
        fitted=U@((eig[:,None]/(eig[:,None]+lam[None,:]))*u1[:,None])
        weights,multiplier=scale_weights(weights_raw,history)
        arrays['returns'][row]=R[t]@weights;arrays['native_loss'][row]=(1-R[t]@weights_raw)**2
        arrays['turnover_proxy'][row]=np.abs(weights-previous).sum(0)
        arrays['complexity'][row]=actual;arrays['lambda'][row]=lam
        arrays['is_sharpe'][row]=sr(fitted);arrays['gross'][row]=np.abs(weights).sum(0)
        if diagnostics:
            cross=K[t,t-window:t]*(history@R[t])
            contribution=((U.T@cross)*u1)[:,None]/(window*(eig[:,None]+lam[None,:]))*multiplier
            contribution=contribution[::-1]
            for band,(a,z) in enumerate([(0,5),(5,15),(15,40),(40,window)]):arrays['contributions'][row,:,band]=contribution[a:z].sum(0)
            arrays['spectral_share'][row]=eig[::-1]/eig.sum();arrays['weights'][row]=weights
        previous=weights
    return dict(arrays,dates=dates[ii],budgets=b,assets=columns,window=window)

def choose(result: dict,cost_bps: float=0.,loss: bool=False) -> int:
    v=result['dates']<=VALIDATION_END
    if v.sum()<36:raise ValueError('Insufficient chronological validation')
    if loss:return int(np.argmin(result['native_loss'][v].mean(0)))
    net=result['returns'][v]-cost_bps/1e4*result['turnover_proxy'][v]
    return int(np.argmax(sr(net)))

def performance(result: dict,index: int,mask: np.ndarray,cost_bps: float=0.) -> dict:
    r=result['returns'][mask,index]-cost_bps/1e4*result['turnover_proxy'][mask,index]
    variance=float(np.var(r,ddof=1)*12);mean=float(np.mean(r)*12);wealth=np.cumprod(1+r)
    dd=wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1
    return dict(n_months=len(r),sharpe=float(sr(r)),mean=mean,volatility=np.sqrt(variance),ce_gamma5=mean-2.5*variance,max_drawdown=float(dd.min()),mean_complexity=float(result['complexity'][mask,index].mean()),mean_lambda=float(result['lambda'][mask,index].mean()),native_loss=float(result['native_loss'][mask,index].mean()),monthly_sleeve_reallocation=float(result['turnover_proxy'][mask,index].mean()),mean_sleeve_gross=float(result['gross'][mask,index].mean()))

def block_bootstrap_differences(a: np.ndarray,b: np.ndarray,*,repetitions: int=1000,block: int=12,seed: int=553) -> dict:
    a=np.asarray(a);b=np.asarray(b)
    if a.ndim==1:a=a[:,None]
    if b.ndim==1:b=b[:,None]
    if len(a)!=len(b):raise ValueError('Unpaired histories')
    rng=np.random.default_rng(seed);n=len(a);vals=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        vals.append(sr(a[idx])-sr(b[idx]))
    vals=np.asarray(vals);observed=sr(a)-sr(b);lo,hi=np.quantile(vals,[.025,.975],axis=0)
    radius=float(np.quantile(np.max(np.abs(vals-observed),axis=1),.95))
    return dict(delta=observed,lo=lo,hi=hi,simultaneous_lo=observed-radius,simultaneous_hi=observed+radius)
