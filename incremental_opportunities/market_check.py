"""Secondary diagnostic prompted by the weak three-tilt baseline.

NOT part of the initial locked protocol. Keep all original results. Add the official
JKP market to the same 57-sleeve master vector and profile four baseline directions.
Keep dates, data vintage cutoff, split, objective, grid and normalization fixed.
No model is selected on this secondary test.
"""
from __future__ import annotations
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from core import fit_extension,BUDGETS,sharpe,paired_inference

URL='https://jkpfactors-data.s3.amazonaws.com/public/%5Busa%5D_%5Bmkt%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip'


def run(out:Path):
    p=out/'inputs';archive=p/'official_jkp_market.zip'
    if not archive.exists():
        response=requests.get(URL,timeout=90);response.raise_for_status();archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as z:
        names=[f for f in z.namelist() if f.endswith('.csv')]
        if len(names)!=1:raise ValueError('Unexpected market ZIP')
        d=pd.read_csv(io.BytesIO(z.read(names[0])),parse_dates=['date'])
    if 'ret' not in d or d.date.duplicated().any():raise ValueError('Unexpected market schema')
    market=d.set_index('date')['ret'].sort_index();market.index=market.index.to_period('M').to_timestamp('M')
    old=pd.read_csv(p/'returns.csv',index_col=0,parse_dates=True)
    panel=pd.concat([market.rename('mkt'),old],axis=1).reindex(old.index)
    if not np.isfinite(panel.to_numpy()).all():raise ValueError('Incomplete matched-date market data')
    panel.to_csv(p/'market_inclusive_returns.csv',index_label='date')
    meta=pd.read_csv(p/'characteristics.csv',index_col=0)
    rms=np.sqrt((panel.loc[:'1994-12-31']**2).mean(0));R=(panel/rms).to_numpy();dd=panel.index
    groups={g:list(range(4))+[i for i,f in enumerate(panel) if i>=4 and meta.loc[f,'group']==g] for g in ['value','investment','momentum']}
    groups['all_extensions']=list(range(panel.shape[1]));rows=[];series={};inference=[]
    for T in [120,240]:
        idx=np.flatnonzero(dd>=pd.Timestamp('1995-01-01'));dates=dd[idx];test=dates>=pd.Timestamp('2005-01-01')
        selected={};baseline=None
        for group,cols in groups.items():
            rr=np.zeros((len(idx),len(BUDGETS)));cc=rr.copy()
            for k,t in enumerate(idx):
                fit=fit_extension(R[t-T:t,cols],4);rr[k]=R[t,cols]@fit.weights;cc[k]=fit.complexity
            if baseline is None:baseline=rr[test,0]
            else:np.testing.assert_allclose(baseline,rr[test,0],atol=1e-9)
            yy=np.zeros(test.sum());complexity=yy.copy()
            for year in range(2005,2026):
                tune=(dates>=pd.Timestamp(year-10,1,1))&(dates<pd.Timestamp(year-5,1,1))
                outer=(dates>=pd.Timestamp(year,1,1))&(dates<pd.Timestamp(year+1,1,1))
                j=int(np.argmin(((1-rr[tune])**2).mean(0)));yy[outer[test]]=rr[outer,j];complexity[outer[test]]=cc[outer,j]
            selected[group]=yy
            rows.append(dict(window=T,model=group,sharpe=float(sharpe(yy)),delta_sharpe=float(sharpe(yy)-sharpe(baseline)),loss_gain=float(np.mean((1-baseline)**2-(1-yy)**2)),mean_C=float(complexity.mean())))
            series[f'{T}__{group}']=yy
        rows.append(dict(window=T,model='baseline',sharpe=float(sharpe(baseline)),delta_sharpe=0.,loss_gain=0.,mean_C=4.))
        series[f'{T}__baseline']=baseline
        for block in [6,12,24]:
            ci=paired_inference(baseline,np.column_stack(list(selected.values())),block=block)
            for j,group in enumerate(selected):inference.append(dict(window=T,model=group,block=block,**{name:float(value[j]) for name,value in ci.items()}))
    pd.DataFrame(rows).to_csv(out/'market_baseline_check.csv',index=False)
    pd.DataFrame(series,index=dates[test]).to_csv(out/'market_check_oos.csv',index_label='date')
    pd.DataFrame(inference).to_csv(out/'market_check_inference.csv',index=False)
    (out/'market_source.json').write_text(json.dumps(dict(url=URL,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scope='Secondary baseline diagnostic; not initial protocol',last_month='2025-12-31'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
