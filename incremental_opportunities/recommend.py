"""Research-only recommendation using completed monthly factor returns.

Usage: python recommend.py --inputs results/inputs --as-of 2025-12-31
Outputs native response-one factor-sleeve exposures, NOT stock orders.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from core import fit_extension,admit_extension,BUDGETS


def recommend(panel:pd.DataFrame, metadata:pd.DataFrame, rms:pd.Series, *,as_of:str, window:int=120) -> dict:
    cutoff=pd.Timestamp(as_of)
    if cutoff<pd.Timestamp('2004-12-31'):raise ValueError('This fixed library is not admitted before 2005')
    panel=panel.loc[:cutoff].copy()
    if panel.empty or panel.index[-1].to_period('M')!=cutoff.to_period('M'):raise ValueError('Missing returns for the requested as-of month')
    if len(panel)<window+120:raise ValueError('Insufficient history for training, tuning and independent confirmation')
    if not panel.index.is_monotonic_increasing or panel.index.duplicated().any():raise ValueError('Unordered or duplicate dates')
    expected=pd.period_range(panel.index.min(),panel.index.max(),freq='M')
    if not panel.index.to_period('M').equals(expected):raise ValueError('Missing calendar months')
    if not np.isfinite(panel.to_numpy()).all():raise ValueError('Missing or nonfinite historical returns')
    if not panel.columns.equals(rms.index) or not panel.columns.equals(metadata.index):raise ValueError('Misaligned input columns')
    if np.any(rms<=0):raise ValueError('Nonpositive calibration')
    R=(panel/rms).to_numpy();names=panel.columns
    groups={g:list(range(3))+[i for i,f in enumerate(names) if i>=3 and metadata.loc[f,'group']==g] for g in sorted(metadata.group.unique())}
    groups={g:c for g,c in groups.items() if len(c)>3};groups['all_extensions']=list(range(len(names)))
    candidates=[];gains=[];decisions=[]
    for group,cols in groups.items():
        r=[];base=[]
        for t in range(len(R)-120,len(R)):
            fit=fit_extension(R[t-window:t,cols],3)
            r.append(R[t,cols]@fit.weights);base.append(R[t,:3]@fit.baseline)
        r=np.array(r);base=np.array(base)
        j=int(np.argmin(((1-r[:60])**2).mean(0)))
        gains.append((1-base[60:])**2-(1-r[60:,j])**2)
        candidates.append((group,cols,j))
        decisions.append(dict(group=group,selected_C_extra=float(BUDGETS[j]),confirmation_loss_gain=float(gains[-1].mean())))
    chosen,lower=admit_extension(np.column_stack(gains),seed=1903+cutoff.year+window)
    w=np.zeros(len(names));group='baseline';C=3.;lam=None
    if chosen>=0:
        group,cols,j=candidates[chosen];f=fit_extension(R[-window:,cols],3)
        w[cols]=f.weights[:,j];C=float(f.complexity[j]);lam=float(f.ridge[j]) if np.isfinite(f.ridge[j]) else None
    else:w[:3]=fit_extension(R[-window:,:3],3,np.array([0.])).baseline
    for d,l in zip(decisions,lower):d['simultaneous_lower_bound']=float(l)
    w=w/rms.to_numpy()
    return dict(status='RESEARCH_PROTOTYPE_NOT_VALIDATED_FOR_DEPLOYMENT',scope='allocations across public characteristic-managed portfolios',as_of=str(panel.index[-1].date()),next_return_month=str((panel.index[-1]+pd.offsets.MonthEnd()).date()),window=window,admitted=group,complexity=C,ridge=lam,tuning_dates=[str(x.date()) for x in panel.index[[-120,-61]]],confirmation_dates=[str(x.date()) for x in panel.index[[-60,-1]]],weights={n:float(v) for n,v in zip(names,w)},candidate_diagnostics=decisions)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--as-of',required=True);p.add_argument('--window',type=int,default=120);p.add_argument('--out',type=Path)
    a=p.parse_args();panel=pd.read_csv(a.inputs/'returns.csv',index_col=0,parse_dates=True);meta=pd.read_csv(a.inputs/'characteristics.csv',index_col=0);rms=pd.read_csv(a.inputs/'fixed_rms.csv',index_col=0).rms
    result=recommend(panel,meta,rms,as_of=a.as_of,window=a.window);text=json.dumps(result,indent=2)
    if a.out:a.out.write_text(text)
    else:print(text)
