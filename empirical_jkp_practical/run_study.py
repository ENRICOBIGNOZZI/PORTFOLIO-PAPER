"""Locked public-JKP study. All choices use validation 1995-2004, not test."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from engine import (prepare,kernel,rolling,choose,performance,sr,scale_weights,
                    block_bootstrap_differences,BASE_BUDGETS,VALIDATION_END)


def run(raw: Path,out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    (out/'paths').mkdir(exist_ok=True)
    data=prepare(raw)
    data.universe_audit.to_csv(out/'universe_audit.csv',index=False)
    data.metadata.to_csv(out/'characteristic_dictionary.csv',index_label='characteristic')
    data.state.to_csv(out/'causal_state.csv',index_label='return_month')
    results={};curves=[];series={};summary=[];t0=time.time()
    def evaluate(name,d,K,n_state=None,**kw):
        t=time.time();res=rolling(d,K,**kw);results[name]=res
        val=res['dates']<=VALIDATION_END;test=~val;j=choose(res)
        v=performance(res,j,val);s=performance(res,j,test)
        row=dict(model=name,n_characteristics=len(res['assets']),n_state=(d.state.shape[1] if n_state is None else n_state),window=res['window'],selected_budget=float(res['budgets'][j]),validation_sharpe=v['sharpe'],**s)
        summary.append(row);series[name]=res['returns'][test,j]
        for i,c in enumerate(res['budgets']):
            vv=performance(res,i,val);tt=performance(res,i,test)
            curves.append(dict(model=name,budget=c,selected=(i==j),validation_sharpe=vv['sharpe'],validation_loss=vv['native_loss'],test_sharpe=tt['sharpe'],test_loss=tt['native_loss'],mean_C=tt['mean_complexity'],mean_lambda=tt['mean_lambda'],mean_IS_Sharpe=float(np.median(res['is_sharpe'][test,i]))))
        pd.DataFrame(dict(return_month=res['dates'],portfolio_return=res['returns'][:,j],complexity=res['complexity'][:,j],lambda_value=res['lambda'][:,j],native_loss=res['native_loss'][:,j],sleeve_reallocation=res['turnover_proxy'][:,j],gross_exposure=res['gross'][:,j])).to_csv(out/'paths'/f'{name}.csv',index=False)
        print(f'{name}: completed in {time.time()-t:.2f}s',flush=True)
        return res,j
    K=kernel(data)
    base,jbase=evaluate('matern_full',data,K,diagnostics=True)
    val=base['dates']<=VALIDATION_END;test=~val
    base_val=float(sr(base['returns'][val,jbase]));base_test=base['returns'][test,jbase]
    pd.DataFrame(base['returns'][test],index=base['dates'][test],columns=base['budgets']).to_csv(out/'primary_path_OOS.csv',index_label='return_month')
    pd.DataFrame(base['is_sharpe'][test],index=base['dates'][test],columns=base['budgets']).to_csv(out/'primary_path_IS.csv',index_label='return_month')
    pd.DataFrame(base['returns'][val],index=base['dates'][val],columns=base['budgets']).to_csv(out/'primary_path_validation.csv',index_label='return_month')
    for kind in ['constant','linear','rbf']:evaluate(kind+'_full',data,kernel(data,kind))
    jloss=choose(base,loss=True)
    summary.append(dict(model='matern_loss_selected',n_characteristics=len(base['assets']),n_state=data.state.shape[1],window=120,selected_budget=float(base['budgets'][jloss]),validation_sharpe=float(sr(base['returns'][val,jloss])),**performance(base,jloss,test)))
    series['matern_loss_selected']=base['returns'][test,jloss]
    R=data.returns.to_numpy(float);ii=np.flatnonzero(data.returns.index.isin(base['dates']))
    equal=[];turn=[];prev=np.zeros(R.shape[1])
    for t in ii:
        w,_=scale_weights(np.ones((R.shape[1],1))/R.shape[1],R[t-120:t]);w=w[:,0]
        equal.append(R[t]@w);turn.append(np.abs(w-prev).sum());prev=w
    er=np.array(equal);series['equal_allocation']=er[test]
    equal_res={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in base.items() if k not in ['weights','contributions','spectral_share']}
    for key in ['returns','native_loss','turnover_proxy','complexity','lambda','is_sharpe','gross']:equal_res[key]=np.zeros((len(er),1))
    equal_res['returns'][:,0]=er;equal_res['turnover_proxy'][:,0]=turn
    for key in ['native_loss','complexity','lambda','gross']:equal_res[key][:,0]=np.nan
    summary.append(dict(model='equal_allocation',n_characteristics=R.shape[1],n_state=0,window=120,selected_budget=np.nan,validation_sharpe=float(sr(er[val])),**performance(equal_res,0,test)))
    pd.DataFrame({'return_month':base['dates'],'return':er,'turnover_proxy':turn}).to_csv(out/'paths'/'equal_allocation.csv',index=False)
    groups=[];group_series=[]
    for g in sorted(data.metadata['group'].unique()):
        assets=list(data.metadata.index[data.metadata['group']!=g])
        res,j=evaluate('drop_sleeves_'+g,data,K,assets=assets)
        match=int(np.argmin(abs(res['budgets']-base['budgets'][jbase])))
        groups.append(dict(group=g,n_removed=R.shape[1]-len(assets),validation_delta_sr=base_val-float(sr(res['returns'][val,j])),test_delta_sr=float(sr(base_test)-sr(res['returns'][test,j])),matched_C_test_delta_sr=float(sr(base_test)-sr(res['returns'][test,match])),selected_C=float(res['complexity'][test,j].mean())))
        group_series.append(res['returns'][test,j])
    gd=pd.DataFrame(groups)
    ci=block_bootstrap_differences(np.repeat(base_test[:,None],len(groups),axis=1),np.column_stack(group_series))
    for name in ['lo','hi','simultaneous_lo','simultaneous_hi']:gd['delta_sr_'+name]=ci[name]
    gd.to_csv(out/'group_ablation.csv',index=False)
    state_rows=[];state_series=[]
    state_specs={g:[c for c in data.state if not c.startswith(g+'__')] for g in sorted(data.metadata['group'].unique())}
    state_specs['all_mean_predictors']=[c for c in data.state if not c.endswith('__mean12')]
    state_specs['all_vol_predictors']=[c for c in data.state if not c.endswith('__vol12')]
    for name,cols in state_specs.items():
        res,j=evaluate('drop_state_'+name,data,kernel(data,state_columns=cols),n_state=len(cols))
        state_rows.append(dict(removed_state=name,n_state_retained=len(cols),validation_delta_sr=base_val-float(sr(res['returns'][val,j])),test_delta_sr=float(sr(base_test)-sr(res['returns'][test,j])),selected_C=res['complexity'][test,j].mean()))
        state_series.append(res['returns'][test,j])
    st=pd.DataFrame(state_rows)
    ci=block_bootstrap_differences(np.repeat(base_test[:,None],len(st),axis=1),np.column_stack(state_series))
    for name in ['lo','hi','simultaneous_lo','simultaneous_hi']:st['delta_sr_'+name]=ci[name]
    st.to_csv(out/'state_ablation.csv',index=False)
    ranking=[]
    for n,f in enumerate(data.returns.columns):
        assets=list(data.returns.columns[data.returns.columns!=f])
        res=rolling(data,K,assets=assets,end=VALIDATION_END);j=choose(res);m=data.metadata.loc[f]
        standalone=data.returns.loc[(data.returns.index>='1995-01-01')&(data.returns.index<=VALIDATION_END),f].to_numpy()
        ranking.append(dict(characteristic=f,description=m['name_new'],group=m['group'],citation=m['cite'],publication_year=m['publication_year'],validation_contribution_sr=base_val-float(sr(res['returns'][:,j])),validation_contribution_sr_matched_C=base_val-float(sr(res['returns'][:,jbase])),validation_standalone_sr=float(sr(standalone)),drop_selected_C=float(res['budgets'][j]),validation_joint_rank=0))
        if (n+1)%25==0:print(f'Validation characteristic ablations {n+1}/{len(data.returns.columns)}',flush=True)
    ranking=pd.DataFrame(ranking).sort_values(['validation_contribution_sr','characteristic'],ascending=[False,True]).reset_index(drop=True)
    ranking['validation_joint_rank']=np.arange(1,len(ranking)+1)
    ranking['validation_standalone_rank']=ranking['validation_standalone_sr'].rank(ascending=False,method='min').astype(int)
    ranking.to_csv(out/'feature_ranking_VALIDATION_ONLY.csv',index=False)
    for n in [8,16,32,64]:
        names=ranking.characteristic.iloc[:n].tolist();sub=prepare(raw,universe=names)
        evaluate(f'compact_top{n}',sub,kernel(sub))
    standalone_names=ranking.sort_values('validation_standalone_sr',ascending=False).characteristic.iloc[:16].tolist()
    sub=prepare(raw,universe=standalone_names);evaluate('standalone_top16',sub,kernel(sub))
    pub=data.metadata.index[data.metadata.publication_year<=2004].tolist()
    sub=prepare(raw,universe=pub);evaluate('published_by2004',sub,kernel(sub))
    finance=['be_me','ni_me','gp_at','op_at','ret_12_1','ret_1_0','at_gr1','oaccruals_at','betabab_1260d','ivol_capm_252d','me','resff3_12_1']
    finance=[f for f in finance if f in data.returns]
    sub=prepare(raw,universe=finance);evaluate('finance_prior',sub,kernel(sub))
    checks=[];check_series=[]
    for f in ranking.characteristic.iloc[:10]:
        res,j=evaluate('drop_feature_'+f,data,K,assets=[a for a in data.returns if a!=f])
        checks.append(dict(characteristic=f,test_delta_sr=float(sr(base_test)-sr(res['returns'][test,j])),matched_C_test_delta_sr=float(sr(base_test)-sr(res['returns'][test,jbase]))))
        check_series.append(res['returns'][test,j])
    check=pd.DataFrame(checks)
    ci=block_bootstrap_differences(np.repeat(base_test[:,None],len(check),axis=1),np.column_stack(check_series))
    for name in ['lo','hi','simultaneous_lo','simultaneous_hi']:check['delta_sr_'+name]=ci[name]
    check.to_csv(out/'top10_feature_holdout_checks.csv',index=False)
    history=[]
    for w in [60,120,240]:
        if w==120:res,j=base,jbase
        else:res,j=evaluate('matern_T'+str(w),data,K,window=w,budgets=np.unique(np.r_[BASE_BUDGETS,w*.985]))
        history.append(dict(window=w,selected_C=res['complexity'][test,j].mean(),mean_lambda=res['lambda'][test,j].mean(),validation_sr=float(sr(res['returns'][val,j])),test_sr=float(sr(res['returns'][test,j]))))
    pd.DataFrame(history).to_csv(out/'history_comparison.csv',index=False)
    nominal=[]
    for p in [16,64,256,1024]:
        res,j=evaluate('matern_rff'+str(p),data,kernel(data,rff_p=p))
        nominal.append(dict(state_features=p,nominal_policy_parameters=p*len(data.returns.columns),selected_C=res['complexity'][test,j].mean(),validation_sr=float(sr(res['returns'][val,j])),test_sr=float(sr(res['returns'][test,j]))))
    pd.DataFrame(nominal).to_csv(out/'nominal_size_comparison.csv',index=False)
    for weighting in ['vw','ew']:
        d=prepare(raw,weighting,universe=list(data.returns.columns));evaluate('matern_'+weighting,d,kernel(d))
    costs=[]
    for cost in [0,10,25,50]:
        j=choose(base,cost_bps=cost)
        costs.append(dict(sleeve_charge_bps=cost,selected_budget=base['budgets'][j],validation_sharpe=float(sr(base['returns'][val,j]-cost/1e4*base['turnover_proxy'][val,j])),**performance(base,j,test,cost_bps=cost)))
    pd.DataFrame(costs).to_csv(out/'cost_PROXY_sensitivity.csv',index=False)
    pd.DataFrame(base['weights'][test,:,jbase],index=base['dates'][test],columns=base['assets']).to_csv(out/'selected_factor_weights.csv',index_label='return_month')
    bands=['ranks_1_5','ranks_6_15','ranks_16_40','ranks_41_120']
    contrib=pd.DataFrame(base['contributions'][test,jbase,:],index=base['dates'][test],columns=bands)
    if not np.allclose(contrib.sum(1).values,base_test,rtol=1e-6,atol=1e-8):raise AssertionError('Contributions do not reconcile')
    contrib.to_csv(out/'oos_spectral_contributions.csv',index_label='return_month')
    pd.DataFrame({'rank':np.arange(1,121),'mean_share_of_training_trace':base['spectral_share'][test].mean(0)}).to_csv(out/'managed_payoff_spectrum.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'selected_models.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'complexity_curves.csv',index=False)
    pd.DataFrame(series,index=base['dates'][test]).to_csv(out/'all_selected_OOS_returns.csv',index_label='return_month')
    boot=[]
    for name in ['constant_full','linear_full','rbf_full','equal_allocation']:
        for block in [6,12,24]:
            ci=block_bootstrap_differences(base_test,series[name],block=block,repetitions=2000)
            boot.append(dict(comparison='matern_full minus '+name,block=block,delta_sr=ci['delta'][0],lo=ci['lo'][0],hi=ci['hi'][0]))
    pd.DataFrame(boot).to_csv(out/'paired_inference.csv',index=False)
    candidates=[s for s in summary if s['model'] in ['compact_top8','compact_top16','compact_top32','compact_top64','matern_full']]
    compact=max(candidates,key=lambda z:z['validation_sharpe'])
    pd.DataFrame([compact]).to_csv(out/'VALIDATION_SELECTED_compact_model.csv',index=False)
    chosen_names=list(data.returns.columns) if compact['model']=='matern_full' else ranking.characteristic.iloc[:int(compact['model'].replace('compact_top',''))].tolist()
    ranking.loc[ranking.characteristic.isin(chosen_names)].to_csv(out/'VALIDATION_SELECTED_characteristics.csv',index=False)
    manifest={'data_source':'Official Jensen-Kelly-Pedersen public USA factor returns','data_level':'characteristic-managed factor portfolios; NOT stock-level CTF','weighting':'capped value weighted','n_factors':data.returns.shape[1],'n_state':data.state.shape[1],'calibration_end':'1994-12-31','validation_start':'1995-01-31','validation_end':VALIDATION_END,'test_start':str(base['dates'][test].min().date()),'test_end':str(base['dates'][test].max().date()),'test_months':int(test.sum()),'selected_compact_model':compact['model'],'validation_characteristic_rank_count':len(ranking),'n_selected_models':len(summary),'n_path_points':len(curves),'seed':739,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'elapsed_seconds':time.time()-t0,'no_stock_level_cost_claim':True,'no_structural_b_r_estimate':True,'source_provenance':json.loads((raw/'provenance.json').read_text())}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FINISHED',json.dumps({k:v for k,v in manifest.items() if k!='source_provenance'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('empirical_jkp_practical/results'))
    a=p.parse_args();run(a.raw,a.out)
