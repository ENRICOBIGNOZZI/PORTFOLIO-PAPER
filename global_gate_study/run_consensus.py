from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import run_global as g


def evaluate_consensus(dataset: str, location: str, weighting: str, T: int, path: dict):
    dd=path['dates']; rr=path['returns']; cc=path['complexity']; cancellation=path['cancellation']
    selectors=['current','gate2','median3','cons_mean','cons_stack','stack3']
    out={k:[] for k in selectors}; C={k:[] for k in selectors}; Z={k:[] for k in selectors}; base=[]; annual=[]
    years=[]
    for year in range(2005,2026):
        old,recent,allp,outer=g.masks(dd,year)
        if old.sum()==60 and recent.sum()==60 and outer.sum()==12: years.append(year)
    if len(years)<3:return [],[]
    for year in years:
        old,recent,allp,outer=g.masks(dd,year)
        lo=g.qloss(rr[old],axis=0); lr=g.qloss(rr[recent],axis=0); la=g.qloss(rr[allp],axis=0)
        jo,jr,ja=int(np.argmin(lo)),int(np.argmin(lr)),int(np.argmin(la))
        jg=min(jo,ja); jm=int(np.median([jo,jr,ja]))
        for name,j in {'current':jo,'gate2':jg,'median3':jm}.items():
            out[name].extend(rr[outer,j]); C[name].extend(cc[outer,j]); Z[name].extend(cancellation[outer,j])
        # Parameter-free consensus midpoint: expansion above Gate2 is limited to a level
        # supported by at least two of old/recent/all selectors.
        out['cons_mean'].extend(.5*rr[outer,jg]+.5*rr[outer,jm])
        C['cons_mean'].extend(.5*cc[outer,jg]+.5*cc[outer,jm])
        Z['cons_mean'].extend(.5*cancellation[outer,jg]+.5*cancellation[outer,jm])
        # Same conservative-to-consensus interval, but choose the mixing weight by the
        # predeployment response-one objective. No tuning constant is introduced.
        wc=g.stack2_weight(rr[allp,jg],rr[allp,jm])
        out['cons_stack'].extend(wc*rr[outer,jg]+(1-wc)*rr[outer,jm])
        C['cons_stack'].extend(wc*cc[outer,jg]+(1-wc)*cc[outer,jm])
        Z['cons_stack'].extend(wc*cancellation[outer,jg]+(1-wc)*cancellation[outer,jm])
        ujs=list(dict.fromkeys([jo,jr,ja])); w3=g.simplex_stack(rr[allp][:,ujs])
        out['stack3'].extend(rr[outer][:,ujs]@w3); C['stack3'].extend(cc[outer][:,ujs]@w3); Z['stack3'].extend(cancellation[outer][:,ujs]@w3)
        base.extend(rr[outer,0])
        annual.append(dict(dataset=dataset,location=location,weighting=weighting,T=T,year=year,
                           C_old=float(g.GRID[jo]),C_recent=float(g.GRID[jr]),C_all=float(g.GRID[ja]),
                           C_gate2=float(g.GRID[jg]),C_consensus=float(g.GRID[jm]),cons_stack_w_gate=float(wc),
                           selector_disagreement=float(max(jo,jr,ja)-min(jo,jr,ja))))
    base=np.asarray(base); rows=[]
    for name in selectors:
        x=np.asarray(out[name]); c=np.asarray(C[name]); z=np.asarray(Z[name]); yp=[]
        for i in range(len(years)):
            sl=slice(12*i,12*(i+1)); yp.append(np.mean((1-base[sl])**2-(1-x[sl])**2)>0)
        rows.append(dict(dataset=dataset,location=location,weighting=weighting,T=T,start_year=years[0],end_year=years[-1],
                         n_years=len(years),n_months=len(x),selector=name,base_sr=g.sr(base),sr=g.sr(x),delta_sr=g.sr(x)-g.sr(base),
                         gainQ=float(np.mean((1-base)**2-(1-x)**2)),meanC=float(c.mean()),sd_monthly_C=float(c.std(ddof=1)),
                         mean_cancellation=float(np.nanmean(z)),p95_cancellation=float(np.nanquantile(z,.95)),positive_year_frac=float(np.mean(yp))))
    return rows,annual


def report(res:pd.DataFrame,annual:pd.DataFrame,skipped:list[dict])->str:
    if res.empty:return '# Consensus Gate stress\n\nNo eligible data.'
    agg=res.groupby('selector').agg(comparisons=('gainQ','size'),datasets=('dataset','nunique'),avg_gainQ=('gainQ','mean'),median_gainQ=('gainQ','median'),
                                     avg_sr=('sr','mean'),avg_delta_sr=('delta_sr','mean'),avg_C=('meanC','mean'),avg_cancel=('mean_cancellation','mean'),positive_year_frac=('positive_year_frac','mean')).sort_values('avg_gainQ',ascending=False)
    cur=res[res.selector.eq('current')][['dataset','T','gainQ','sr']].rename(columns={'gainQ':'g0','sr':'s0'})
    x=res.merge(cur,on=['dataset','T']); wins=[]
    for name,z in x.groupby('selector'):
        wins.append(dict(selector=name,q_wins=int((z.gainQ>z.g0+1e-12).sum()),q_losses=int((z.gainQ<z.g0-1e-12).sum()),sr_wins=int((z.sr>z.s0+1e-12).sum()),sr_losses=int((z.sr<z.s0-1e-12).sum())))
    wins=pd.DataFrame(wins).set_index('selector')
    return '\n'.join(['# Conservative-to-consensus Gate stress','',f'Eligible comparisons: **{cur.shape[0]}** across **{res.dataset.nunique()} datasets**.','',
                      'The factor library is the previously frozen 40-factor set. `cons_mean` and `cons_stack` never expand beyond the median of old/recent/all predeployment complexity choices, so additional complexity requires support from at least two temporal views.','',
                      '## Aggregate','', '```',agg.round(4).to_string(),'```','', '## Wins/losses vs current','', '```',wins.to_string(),'```','',
                      '## Annual diagnostics','',f'Mean Gate2-to-consensus grid gap: {(annual.C_consensus-annual.C_gate2).mean():.3f}.',f'Mean cons-stack weight on Gate2: {annual.cons_stack_w_gate.mean():.3f}.','',
                      '## Skipped','', '```',pd.DataFrame(skipped).to_string(index=False) if skipped else 'None','```'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--locations',nargs='*',default=g.DEFAULT_LOCATIONS);a=ap.parse_args()
    a.raw.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    datasets=[]
    for loc in a.locations:
        p=g.download(a.raw,loc,'vw_cap')
        if p is not None:datasets.append((f'{loc}_vwcap',loc,'vw_cap',p))
    for wt in ['vw','ew']:
        p=g.download(a.raw,'usa',wt)
        if p is not None:datasets.append((f'usa_{wt}','usa',wt,p))
    rows=[]; annual=[]; skipped=[]; manifest=[]
    for name,loc,wt,path in datasets:
        try:
            p=g.load_zip(path); missing=[c for c in g.COMMON40 if c not in p.columns]
            if missing: skipped.append(dict(dataset=name,reason=f'missing {len(missing)} frozen factors'));continue
            usable=g.contiguous_suffix(p[g.COMMON40])
            if len(usable)<240: skipped.append(dict(dataset=name,reason=f'usable suffix only {len(usable)} months'));continue
            manifest.append(dict(dataset=name,first=usable.index.min().date(),last=usable.index.max().date(),months=len(usable)))
            for T in [60,120,240]:
                try:
                    path_data=g.make_path(usable,T); r,aa=evaluate_consensus(name,loc,wt,T,path_data)
                    if not r: skipped.append(dict(dataset=f'{name}_T{T}',reason='fewer than 3 deployment years'));continue
                    rows.extend(r);annual.extend(aa)
                except Exception as exc: skipped.append(dict(dataset=f'{name}_T{T}',reason=repr(exc)))
            print('DONE',name,flush=True)
        except Exception as exc: skipped.append(dict(dataset=name,reason=repr(exc)))
    res=pd.DataFrame(rows);ann=pd.DataFrame(annual)
    res.to_csv(a.out/'consensus_selector_results.csv',index=False);ann.to_csv(a.out/'consensus_annual.csv',index=False);pd.DataFrame(skipped).to_csv(a.out/'consensus_skipped.csv',index=False);pd.DataFrame(manifest).to_csv(a.out/'consensus_manifest.csv',index=False)
    if not res.empty:
        res.groupby('selector').agg(comparisons=('gainQ','size'),datasets=('dataset','nunique'),avg_gainQ=('gainQ','mean'),median_gainQ=('gainQ','median'),avg_sr=('sr','mean'),avg_delta_sr=('delta_sr','mean'),avg_C=('meanC','mean'),positive_year_frac=('positive_year_frac','mean')).reset_index().to_csv(a.out/'consensus_aggregate.csv',index=False)
    (a.out/'CONSENSUS_REPORT.md').write_text(report(res,ann,skipped));print((a.out/'CONSENSUS_REPORT.md').read_text())

if __name__=='__main__':main()
