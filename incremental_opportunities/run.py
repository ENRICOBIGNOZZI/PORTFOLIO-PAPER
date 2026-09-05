"""Run fixed-universe, nested opportunity learning from an archived JKP snapshot."""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from core import fit_extension, BUDGETS, sharpe, admit_extension, paired_inference, population_metrics

ANCHORS=['market_equity','be_me','ret_12_1']


def load_data(raw:Path,out:Path):
    factor_path=next((raw/'usa_all_factors_vw_cap').glob('*.csv'))
    d=pd.read_csv(factor_path,parse_dates=['date'])
    meta=pd.read_csv(raw/'factor_details.csv').dropna(subset=['abr_jkp']).drop_duplicates('abr_jkp').set_index('abr_jkp')
    meta['publication_year']=pd.to_numeric(meta['cite'].str.extract(r'((?:19|20)\d{2})')[0],errors='coerce')
    panel=d.pivot(index='date',columns='name',values='ret').sort_index().loc['1973-01-01':'2025-12-31']
    panel.index=panel.index.to_period('M').to_timestamp('M')
    panel=panel.reindex(pd.date_range(panel.index.min(),panel.index.max(),freq=pd.offsets.MonthEnd()))
    candidates=[f for f in panel if f in meta.index and meta.loc[f,'publication_year']<=2004]
    eligible=[f for f in candidates if panel.loc[:'1994-12-31',f].notna().all() and panel.loc[:'1994-12-31',f].std()>1e-8]
    names=ANCHORS+[f for f in sorted(eligible) if f not in ANCHORS]
    panel=panel[names];meta=meta.loc[names].copy()
    if panel.isna().any().any():raise ValueError('Incomplete fixed panel: needs explicit missing-payoff rule')
    meta['group']=meta.group.str.lower().str.replace(' ','_',regex=False)
    rms=np.sqrt((panel.loc[:'1994-12-31']**2).mean(0))
    if np.any(rms<1e-8):raise ValueError('Invalid calibration')
    panel.to_csv(out/'inputs'/'returns.csv',index_label='date')
    meta.to_csv(out/'inputs'/'characteristics.csv',index_label='characteristic')
    rms.to_csv(out/'inputs'/'fixed_rms.csv',header=['rms'],index_label='characteristic')
    sources={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [factor_path,raw/'factor_details.csv']}
    return panel,meta,rms,sources


def paths(R:np.ndarray,dates:pd.DatetimeIndex,cols:list[int],T:int,budgets=BUDGETS):
    idx=np.flatnonzero((dates>=pd.Timestamp('1995-01-01'))&(np.arange(len(R))>=T))
    ret=np.empty((len(idx),len(budgets)));unhedged=ret.copy();Cs=ret.copy();lam=ret.copy()
    base=np.empty(len(idx));orders=np.zeros((len(idx),len(cols),len(budgets)))
    for s,t in enumerate(idx):
        f=fit_extension(R[t-T:t,cols],3,np.asarray(budgets))
        ret[s]=R[t,cols]@f.weights;unhedged[s]=R[t,cols]@f.unhedged_weights
        base[s]=R[t,:3]@f.baseline;Cs[s]=f.complexity;lam[s]=f.ridge;orders[s]=f.weights
    if budgets[0]==0 and not np.allclose(ret[:,0],base,atol=1e-9):raise AssertionError('Failed exact baseline nesting')
    return dict(dates=dates[idx],returns=ret,unhedged=unhedged,base=base,complexity=Cs,ridge=lam,weights=orders,cols=cols)


def run_empirical(raw:Path,out:Path):
    (out/'inputs').mkdir(parents=True,exist_ok=True)
    panel,meta,rms,sources=load_data(raw,out);R=(panel/rms).to_numpy();dates=panel.index
    group_cols={g:list(range(3))+[i for i,f in enumerate(panel) if i>=3 and meta.loc[f,'group']==g] for g in sorted(meta.group.unique())}
    group_cols={g:c for g,c in group_cols.items() if len(c)>3};group_cols['all_extensions']=list(range(len(panel.columns)))
    summaries=[];admissions=[];choices=[];year_results=[];all_series={};fixed_rows=[];last_weights=[];boot_rows=[]
    for T in [60,120,240]:
        cache={g:paths(R,dates,c,T) for g,c in group_cols.items()}
        first=next(iter(cache.values()));dd=first['dates'];test=dd>=pd.Timestamp('2005-01-01');n=int(test.sum())
        model_names=['baseline']+list(cache)+['admission_gate','unhedged_all_extensions']
        deployed={m:np.zeros(n) for m in model_names};comp={m:np.full(n,3.) for m in model_names}
        deployed['baseline']=first['base'][test]
        for yr in range(2005,2026):
            tune=(dd>=pd.Timestamp(yr-10,1,1))&(dd<pd.Timestamp(yr-5,1,1))
            confirm=(dd>=pd.Timestamp(yr-5,1,1))&(dd<pd.Timestamp(yr,1,1))
            outer=(dd>=pd.Timestamp(yr,1,1))&(dd<pd.Timestamp(yr+1,1,1));oi=outer[test]
            if tune.sum()!=60 or confirm.sum()!=60 or outer.sum()!=12:raise AssertionError('Chronology split')
            gain=[];chosen=[];names=list(cache)
            for g,res in cache.items():
                j=int(np.argmin(((1-res['returns'][tune])**2).mean(0)));chosen.append(j)
                gain.append((1-res['base'][confirm])**2-(1-res['returns'][confirm,j])**2)
                deployed[g][oi]=res['returns'][outer,j];comp[g][oi]=res['complexity'][outer,j]
                choices.append(dict(year=yr,window=T,group=g,tune_end=f'{yr-6}-12-31',confirmation_end=f'{yr-1}-12-31',selected_grid_C_extra=float(BUDGETS[j]),training_total_C=float(res['complexity'][outer,j].mean()),confirmation_loss_gain=float(np.mean(gain[-1]))))
                if g=='all_extensions':
                    deployed['unhedged_all_extensions'][oi]=res['unhedged'][outer,j];comp['unhedged_all_extensions'][oi]=res['complexity'][outer,j]
            which,lower=admit_extension(np.column_stack(gain),seed=1903+yr+T)
            label=names[which] if which>=0 else 'baseline'
            deployed['admission_gate'][oi]=deployed[label][oi];comp['admission_gate'][oi]=comp[label][oi]
            admissions.append(dict(year=yr,window=T,admitted=label,lower_bound=float(lower[which]) if which>=0 else float(lower.max()),selected_C_extra=float(BUDGETS[chosen[which]]) if which>=0 else 0.))
            for g,l in zip(names,lower):choices[-len(names)+names.index(g)]['simultaneous_lower_bound']=float(l)
            for name in deployed:
                rr=deployed[name][oi];bb=deployed['baseline'][oi]
                year_results.append(dict(year=yr,window=T,model=name,mean_loss=float(np.mean((1-rr)**2)),incremental_loss_gain=float(np.mean((1-bb)**2-(1-rr)**2)),annualized_mean=float(12*rr.mean()),mean_C=float(comp[name][oi].mean())))
        for name,y in deployed.items():
            summaries.append(dict(window=T,model=name,n_months=n,sharpe=float(sharpe(y)),mean=float(y.mean()*12),volatility=float(y.std(ddof=1)*np.sqrt(12)),response_one_loss=float(np.mean((1-y)**2)),delta_sharpe=float(sharpe(y)-sharpe(deployed['baseline'])),loss_gain=float(np.mean((1-deployed['baseline'])**2-(1-y)**2)),mean_C=float(comp[name].mean())))
            all_series[f'{T}__{name}']=y
        for g,res in cache.items():
            for j,c in enumerate(BUDGETS):
                fixed_rows.append(dict(window=T,group=g,budget_extra=c,mean_C=res['complexity'][test,j].mean(),sharpe=float(sharpe(res['returns'][test,j])),delta_sharpe=float(sharpe(res['returns'][test,j])-sharpe(res['base'][test])),loss_gain=float(np.mean((1-res['base'][test])**2-(1-res['returns'][test,j])**2)),is_loss=float('nan')))
        names=[g for g in deployed if g!='baseline']
        for block in [6,12,24]:
            ci=paired_inference(deployed['baseline'],np.column_stack([deployed[g] for g in names]),block=block)
            for k,g in enumerate(names):boot_rows.append(dict(window=T,model=g,block=block,**{key:float(vals[k]) for key,vals in ci.items()}))
        selected={z['group']:z['selected_grid_C_extra'] for z in choices if z['year']==2025 and z['window']==T}
        for g,res in cache.items():
            j=int(np.where(BUDGETS==selected[g])[0][0]);ww=res['weights'][-1,:,j]/rms.iloc[res['cols']].to_numpy()
            for name,w in zip(panel.columns[res['cols']],ww):last_weights.append(dict(return_month='2025-12-31',window=T,group=g,characteristic=name,native_weight=w))
        print('Finished empirical window',T,flush=True)
    for name,rows in [('summary',summaries),('admission_log',admissions),('selection_log',choices),('yearly_results',year_results),('fixed_complexity_paths',fixed_rows),('paired_inference',boot_rows),('historical_weights_dec2025',last_weights)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame(all_series,index=dd[test]).to_csv(out/'oos_returns.csv',index_label='return_month')
    small=[];small_series=[];base=None
    for f in panel.columns[3:]:
        cols=[0,1,2,panel.columns.get_loc(f)];res=paths(R,dates,cols,120,np.array([0.,.5]));tt=res['dates']>=pd.Timestamp('2005-01-01')
        a=res['base'][tt];b=res['returns'][tt,1];base=a;small_series.append(b);y=panel.loc[res['dates'][tt],f].to_numpy()
        small.append(dict(characteristic=f,description=meta.loc[f,'name_new'],group=meta.loc[f,'group'],publication_year=meta.loc[f,'publication_year'],standalone_SR=float(sharpe(y)),paired_delta_SR=float(sharpe(b)-sharpe(a)),loss_gain=float(np.mean((1-a)**2-(1-b)**2))))
    ci=paired_inference(base,np.column_stack(small_series))
    for i,row in enumerate(small):
        for key,values in ci.items():row[key]=float(values[i])
    pd.DataFrame(small).to_csv(out/'individual_diagnostics_not_selection.csv',index=False)
    manifest=dict(source='Official JKP archived public characteristic-managed portfolios',original_sha256=sources,n_factors=len(panel.columns),baseline=ANCHORS,n_months=252,library_publication_cutoff=2004,calibration_end='1994-12-31',tuning_months=60,confirmation_months=60,outer_months=12,test_start='2005-01-31',test_end='2025-12-31',exploratory_reused_history=True,nested_classes=True,state_dimension=0,no_stock_level_claim=True,finite_rank_only=True,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


def run_simulation(out:Path,reps:int=400):
    p=24;n0=3;d=.03**2*np.r_[np.array([1.,.8,.6]),np.arange(1,p+1,dtype=float)**-2]
    D=np.diag(d);q0=.7**2/(12+.7**2);q1=1./13-q0
    budgets=np.array([0.,.5,1.,2.,4.,8.,12.,16.,20.,23.]);means={}
    for mode in ['leading','trailing']:
        v=np.zeros(n0+p);v[:3]=np.sqrt(q0/3)
        loc=np.arange(3,7) if mode=='leading' else np.arange(3+p-4,3+p)
        v[loc]=np.sqrt(q1/4);means[mode]=np.sqrt(d)*v
    cov={mode:D-np.outer(m,m) for mode,m in means.items()};chol={mode:np.linalg.cholesky(S) for mode,S in cov.items()}
    records=[];laws=[]
    for T in [60,120,240,480,960]:
        for rep in range(reps):
            rng=np.random.default_rng(81009+100000*T+rep);Z=rng.normal(size=(T+120,n0+p))
            for mode in means:
                m=means[mode];R=m+Z@chol[mode].T;fit=fit_extension(R[:T],n0,budgets)
                q,sr=population_metrics(fit.weights,m,D);val=((1-R[T:]@fit.weights)**2).mean(0)
                j=int(np.argmin(val));baseq=q[0];basesr=sr[0]
                for k,c in enumerate(budgets):laws.append(dict(T=T,rep=rep,signal=mode,C_extra=float(c),C=fit.complexity[k],lambda_value=fit.ridge[k],population_Q=q[k],population_SR=sr[k],incremental_Q_recovery=(baseq-q[k])/q1,delta_SR=sr[k]-basesr))
                records.append(dict(T=T,rep=rep,signal=mode,selected_C=float(fit.complexity[j]),population_Q=float(q[j]),population_SR=float(sr[j]),incremental_Q_recovery=float((baseq-q[j])/q1),delta_SR=float(sr[j]-basesr),positive_gain=bool(baseq>q[j]+1e-12),baseline_SR=float(basesr),baseline_Q=float(baseq)))
        print('Finished known-economy simulation',T,flush=True)
    diag=pd.DataFrame(records);curves=pd.DataFrame(laws)
    diag.to_csv(out/'simulation_replications.csv',index=False);curves.to_csv(out/'simulation_path_replications.csv',index=False)
    diag.groupby(['T','signal']).agg(selected_C=('selected_C','mean'),population_SR=('population_SR','mean'),incremental_Q_recovery=('incremental_Q_recovery','mean'),delta_SR=('delta_SR','mean'),probability_positive=('positive_gain','mean'),recovery_se=('incremental_Q_recovery',lambda x:x.std(ddof=1)/np.sqrt(len(x)))).to_csv(out/'simulation_summary.csv')
    curves.groupby(['T','signal','C_extra']).agg(population_Q=('population_Q','mean'),population_SR=('population_SR','mean'),incremental_Q_recovery=('incremental_Q_recovery','mean'),delta_SR=('delta_SR','mean')).to_csv(out/'simulation_curves.csv')
    oracle={mode:dict(baseline_SR=.7,full_SR=1.,baseline_Q=1-q0,full_Q=1-q0-q1,incremental_Q=q1,second_moment_diagonal=d.tolist(),mean=m.tolist(),target_norm=float(np.linalg.norm(m/d))) for mode,m in means.items()}
    (out/'simulation_oracles.json').write_text(json.dumps(oracle,indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--out',type=Path,default=Path('results'));a.add_argument('--reps',type=int,default=400)
    args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True);t=time.time()
    manifest=run_empirical(args.raw,args.out);run_simulation(args.out,args.reps)
    manifest['simulation_reps']=args.reps;manifest['elapsed_seconds']=time.time()-t
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',manifest['elapsed_seconds'])
