from __future__ import annotations

import argparse, io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

import run_global as g


def download_market(raw:Path,location:str)->Path|None:
    raw.mkdir(parents=True,exist_ok=True)
    s=f'[{location}]_[mkt]_[monthly]_[vw_cap]'; path=raw/f'{s}.zip'
    if path.exists() and path.stat().st_size>100:return path
    url=g.BASE_URL+'/'+s.replace('[','%5B').replace(']','%5D')+'.zip'
    try:
        r=requests.get(url,timeout=(10,90))
        if r.status_code!=200 or len(r.content)<100:
            print('MKT UNAVAILABLE',location,r.status_code,flush=True);return None
        path.write_bytes(r.content);return path
    except Exception as exc:
        print('MKT FAILED',location,repr(exc),flush=True);return None


def load_market(path:Path)->pd.Series:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('.csv')]
        if len(names)!=1:raise ValueError('unexpected market archive')
        d=pd.read_csv(io.BytesIO(z.read(names[0])),parse_dates=['date'])
    if 'ret' not in d.columns:raise ValueError('market ret column missing')
    s=d.set_index('date')['ret'].sort_index();s.index=s.index.to_period('M').to_timestamp('M');return s


def make_path4(P:pd.DataFrame,T:int)->dict:
    if len(P)<max(96,T+120+36):raise ValueError('insufficient history')
    rms=np.sqrt((P.iloc[:96]**2).mean(0)).replace(0,np.nan)
    if rms.isna().any() or (rms<1e-12).any():raise ValueError('degenerate RMS')
    R=(P/rms).to_numpy(float);idx=np.arange(T,len(P));rr=np.empty((len(idx),len(g.GRID)));cc=rr.copy();zz=rr.copy();ranks=np.empty(len(idx),int)
    for n,t in enumerate(idx):
        f=g.fit_gate(R[t-T:t],4);rr[n]=R[t]@f.weights;cc[n]=f.c_total;zz[n]=f.cancellation;ranks[n]=f.numerical_rank
    return dict(dates=P.index[idx],returns=rr,complexity=cc,cancellation=zz,rank=ranks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--locations',nargs='*',default=g.DEFAULT_LOCATIONS);a=ap.parse_args()
    a.raw.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    rows=[];annual=[];skipped=[];manifest=[]
    for loc in a.locations:
        pf=g.download(a.raw,loc,'vw_cap');pm=download_market(a.raw,loc)
        if pf is None or pm is None:skipped.append(dict(dataset=loc,reason='missing factor or market archive'));continue
        try:
            factors=g.load_zip(pf);missing=[x for x in g.COMMON40 if x not in factors.columns]
            if missing:skipped.append(dict(dataset=loc,reason=f'missing {len(missing)} frozen factors'));continue
            panel=pd.concat([load_market(pm).rename('mkt'),factors[g.COMMON40]],axis=1)
            usable=g.contiguous_suffix(panel)
            if len(usable)<240:skipped.append(dict(dataset=loc,reason=f'usable suffix {len(usable)} months'));continue
            name=f'{loc}_market_vwcap';manifest.append(dict(dataset=name,first=usable.index.min().date(),last=usable.index.max().date(),months=len(usable)))
            for T in [60,120,240]:
                try:
                    p=make_path4(usable,T);r,aa=g.evaluate(name,loc,'vw_cap',T,p)
                    if not r:skipped.append(dict(dataset=f'{name}_T{T}',reason='fewer than 3 deployment years'));continue
                    rows.extend(r);annual.extend(aa)
                except Exception as exc:skipped.append(dict(dataset=f'{name}_T{T}',reason=repr(exc)))
            print('DONE MARKET',loc,flush=True)
        except Exception as exc:skipped.append(dict(dataset=loc,reason=repr(exc)))
    res=pd.DataFrame(rows);ann=pd.DataFrame(annual)
    res.to_csv(a.out/'market_global_selector_results.csv',index=False);ann.to_csv(a.out/'market_global_annual.csv',index=False);pd.DataFrame(skipped).to_csv(a.out/'market_global_skipped.csv',index=False);pd.DataFrame(manifest).to_csv(a.out/'market_global_manifest.csv',index=False)
    if not res.empty:
        agg=res.groupby('selector').agg(comparisons=('gainQ','size'),datasets=('dataset','nunique'),avg_gainQ=('gainQ','mean'),median_gainQ=('gainQ','median'),avg_sr=('sr','mean'),avg_delta_sr=('delta_sr','mean'),avg_C=('meanC','mean'),positive_year_frac=('positive_year_frac','mean')).sort_values('avg_gainQ',ascending=False)
        cur=res[res.selector.eq('current')][['dataset','T','gainQ','sr']].rename(columns={'gainQ':'g0','sr':'s0'});x=res.merge(cur,on=['dataset','T']);wins=[]
        for sel,z in x.groupby('selector'):wins.append(dict(selector=sel,q_wins=int((z.gainQ>z.g0+1e-12).sum()),q_losses=int((z.gainQ<z.g0-1e-12).sum()),sr_wins=int((z.sr>z.s0+1e-12).sum()),sr_losses=int((z.sr<z.s0-1e-12).sum())))
        wins=pd.DataFrame(wins).set_index('selector')
        txt='\n'.join(['# Global market-inclusive Gate stress','',f'Eligible comparisons: **{len(cur)}** across **{res.dataset.nunique()} datasets**.','',
                       'Each baseline contains the official local JKP market return plus size, book-to-market, and 12-1 momentum. The same frozen 37 residual extensions are then added.','',
                       '## Aggregate','', '```',agg.round(4).to_string(),'```','', '## Wins/losses vs current','', '```',wins.to_string(),'```','',
                       '## Skipped','', '```',pd.DataFrame(skipped).to_string(index=False) if skipped else 'None','```'])
    else:txt='# Global market-inclusive Gate stress\n\nNo eligible data.'
    (a.out/'MARKET_GLOBAL_REPORT.md').write_text(txt);print(txt)

if __name__=='__main__':main()
