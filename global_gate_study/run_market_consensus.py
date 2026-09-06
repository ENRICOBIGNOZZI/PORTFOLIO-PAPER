from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import run_global as g
import run_consensus as c
import run_market_baseline as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--locations',nargs='*',default=g.DEFAULT_LOCATIONS);a=ap.parse_args()
    a.raw.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    rows=[];annual=[];skipped=[];manifest=[]
    for loc in a.locations:
        pf=g.download(a.raw,loc,'vw_cap');pm=m.download_market(a.raw,loc)
        if pf is None or pm is None:skipped.append(dict(dataset=loc,reason='missing factor or market archive'));continue
        try:
            factors=g.load_zip(pf);missing=[x for x in g.COMMON40 if x not in factors.columns]
            if missing:skipped.append(dict(dataset=loc,reason=f'missing {len(missing)} frozen factors'));continue
            panel=pd.concat([m.load_market(pm).rename('mkt'),factors[g.COMMON40]],axis=1);usable=g.contiguous_suffix(panel)
            if len(usable)<240:skipped.append(dict(dataset=loc,reason=f'usable suffix {len(usable)} months'));continue
            name=f'{loc}_market_vwcap';manifest.append(dict(dataset=name,first=usable.index.min().date(),last=usable.index.max().date(),months=len(usable)))
            for T in [60,120,240]:
                try:
                    p=m.make_path4(usable,T);r,aa=c.evaluate_consensus(name,loc,'vw_cap',T,p)
                    if not r:skipped.append(dict(dataset=f'{name}_T{T}',reason='fewer than 3 deployment years'));continue
                    rows.extend(r);annual.extend(aa)
                except Exception as exc:skipped.append(dict(dataset=f'{name}_T{T}',reason=repr(exc)))
            print('DONE',name,flush=True)
        except Exception as exc:skipped.append(dict(dataset=loc,reason=repr(exc)))
    res=pd.DataFrame(rows);ann=pd.DataFrame(annual)
    res.to_csv(a.out/'market_consensus_results.csv',index=False);ann.to_csv(a.out/'market_consensus_annual.csv',index=False);pd.DataFrame(skipped).to_csv(a.out/'market_consensus_skipped.csv',index=False);pd.DataFrame(manifest).to_csv(a.out/'market_consensus_manifest.csv',index=False)
    txt=c.report(res,ann,skipped).replace('# Conservative-to-consensus Gate stress','# Market-inclusive conservative-to-consensus Gate stress').replace('The factor library is the previously frozen 40-factor set.','Each baseline contains the official local JKP market plus size, book-to-market and 12-1 momentum; the residual library is frozen.')
    (a.out/'MARKET_CONSENSUS_REPORT.md').write_text(txt);print(txt)

if __name__=='__main__':main()
