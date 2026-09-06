from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from engine import prepare, rolling, performance, sr, VALIDATION_END

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'jkp_public_raw_20260906'
if not RAW.exists():
    RAW = ROOT.parent / 'jkp_public_raw_20260906'
OUT = ROOT / 'results_fixed_matern_20260906'
OUT.mkdir(parents=True, exist_ok=True)

ELL_GRID = np.array([0.125,0.18,0.25,0.35,0.5,0.7,1.0,1.4,2.0,2.8,4.0,5.6,8.0,11.2,16.0,22.6,32.0,45.0,64.0])
C_GRID = np.array([0.25,0.5,0.75,1,1.5,2,3,4,6,8,10,12,16,20,24,28,32,40,48,56,64,72,80,88,96,104,112,118.0])
HISTORY_GRID = [60,120,180,240]

def matern32(data, ell_mult: float) -> np.ndarray:
    x=(data.state.to_numpy(float)-data.calibration_mean)/data.calibration_sd
    d=cdist(x,x)/(data.ell*ell_mult)
    u=np.sqrt(3.0)*d
    return (1.0+u)*np.exp(-u)

def summarize(res, ell_mult: float) -> pd.DataFrame:
    val=res['dates']<=VALIDATION_END
    test=~val
    rows=[]
    for j,c in enumerate(res['budgets']):
        pv=performance(res,j,val)
        pt=performance(res,j,test)
        rows.append(dict(ell_mult=ell_mult,budget=float(c),validation_loss=pv['native_loss'],validation_sharpe=pv['sharpe'],test_loss=pt['native_loss'],test_sharpe=pt['sharpe'],test_mean=pt['mean'],test_volatility=pt['volatility'],mean_lambda=pt['mean_lambda'],mean_complexity=pt['mean_complexity'],median_is_sharpe=float(np.median(res['is_sharpe'][test,j]))))
    return pd.DataFrame(rows)

def select_by_validation_loss(grid: pd.DataFrame) -> pd.Series:
    return grid.sort_values(['validation_loss','ell_mult','budget']).iloc[0]

def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',dpi=260,bbox_inches='tight')
    plt.close(fig)

def main():
    data=prepare(RAW)
    all_rows=[]
    for ell in ELL_GRID:
        K=matern32(data,float(ell))
        res=rolling(data,K,window=120,budgets=C_GRID)
        tab=summarize(res,float(ell))
        all_rows.append(tab)
        print(f'ell={ell:g}: best validation loss {tab.validation_loss.min():.6f}',flush=True)
    grid=pd.concat(all_rows,ignore_index=True)
    grid.to_csv(OUT/'hyperparameter_grid.csv',index=False)
    chosen=select_by_validation_loss(grid)
    ell_star=float(chosen.ell_mult); c_star=float(chosen.budget)
    print('SELECTED',chosen.to_dict(),flush=True)

    Kstar=matern32(data,ell_star)
    main_res=rolling(data,Kstar,window=120,budgets=C_GRID,diagnostics=True)
    path=summarize(main_res,ell_star)
    path['selected']=np.isclose(path.budget,c_star)
    test_min=float(path.loc[path.test_loss.idxmin(),'budget'])
    path['test_loss_min']=np.isclose(path.budget,test_min)
    path.to_csv(OUT/'complexity_path.csv',index=False)
    val=main_res['dates']<=VALIDATION_END; test=~val
    jstar=int(np.argmin(np.abs(main_res['budgets']-c_star)))
    selected_perf=performance(main_res,jstar,test); selected_val=performance(main_res,jstar,val)
    selected=dict(ell_mult=ell_star,budget=c_star,validation_loss=selected_val['native_loss'],validation_sharpe=selected_val['sharpe'],test_loss=selected_perf['native_loss'],test_sharpe=selected_perf['sharpe'],test_mean=selected_perf['mean'],test_volatility=selected_perf['volatility'],mean_lambda=selected_perf['mean_lambda'],mean_complexity=selected_perf['mean_complexity'])

    history=[]
    for w in HISTORY_GRID:
        b=np.unique(np.minimum(C_GRID,w*.985))
        r=rolling(data,Kstar,window=w,budgets=b)
        v=r['dates']<=VALIDATION_END; t=~v
        losses=r['native_loss'][v].mean(0); j=int(np.argmin(losses))
        history.append(dict(window=w,selected_C=float(r['budgets'][j]),validation_loss=float(losses[j]),test_loss=float(r['native_loss'][t,j].mean()),test_sharpe=float(sr(r['returns'][t,j])),mean_lambda=float(r['lambda'][t,j].mean())))
    history=pd.DataFrame(history); history.to_csv(OUT/'history_path.csv',index=False)
    shares=main_res['spectral_share'][test]
    spectrum=pd.DataFrame({'rank':np.arange(1,shares.shape[1]+1),'mean_share':shares.mean(0),'p10':np.quantile(shares,.10,axis=0),'p90':np.quantile(shares,.90,axis=0)})
    spectrum.to_csv(OUT/'managed_payoff_spectrum.csv',index=False)

    fig,ax=plt.subplots(figsize=(6.2,4.1)); ax.plot(path.mean_complexity,path.validation_loss,label='Validation loss'); ax.plot(path.mean_complexity,path.test_loss,label='Test loss')
    s=path.loc[path.selected].iloc[0]; m=path.loc[path.test_loss_min].iloc[0]
    ax.scatter([s.mean_complexity],[s.test_loss],marker='D',s=45,label='Validation-selected'); ax.scatter([m.mean_complexity],[m.test_loss],facecolors='none',edgecolors='black',s=55,label='Ex post test minimum')
    ax.set_xlabel('Managed-payoff effective dimension'); ax.set_ylabel('Response-one loss'); ax.set_title('One fixed Matérn-3/2 representation: loss across complexity'); ax.legend(frameon=False); savefig(fig,'fig_emp_01_loss_complexity')

    fig,ax=plt.subplots(figsize=(6.2,4.1)); ax.plot(path.mean_complexity,path.median_is_sharpe,label='Median in-sample Sharpe'); ax.plot(path.mean_complexity,path.test_sharpe,label='Out-of-sample Sharpe'); ax.axvline(c_star,ls=':',lw=1,label='Validation-selected complexity')
    ax.set_xlabel('Managed-payoff effective dimension'); ax.set_ylabel('Sharpe ratio'); ax.set_title('Fit is not learnability'); ax.legend(frameon=False); savefig(fig,'fig_emp_02_is_oos_sharpe')

    fig,ax=plt.subplots(figsize=(6.2,4.1)); x=spectrum['rank'].to_numpy(); ax.plot(x,np.maximum(spectrum['mean_share'],1e-8),label='Mean normalized eigenvalue'); ax.fill_between(x,np.maximum(spectrum['p10'],1e-8),np.maximum(spectrum['p90'],1e-8),alpha=.18,label='10--90% range')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Ordered managed-payoff direction'); ax.set_ylabel('Share of trace'); ax.set_title('Managed-payoff spectrum in the fixed Matérn representation'); ax.legend(frameon=False); savefig(fig,'fig_emp_03_spectrum')

    fig,ax=plt.subplots(figsize=(6.2,4.1)); ax.plot(history.window,history.selected_C,marker='o'); ax.set_xlabel('Training history T (months)'); ax.set_ylabel('Validation-selected effective dimension'); ax.set_title('More history changes the data-supported decision dimension'); savefig(fig,'fig_emp_04_history_complexity')

    points=[path.iloc[int(np.argmin(np.abs(path.budget-target)))] for target in [4.0,c_star,118.0]]
    fig,ax=plt.subplots(figsize=(6.2,4.1)); sigma=np.linspace(0,.15,100); labels=['Under-activated','Validation-selected','Near interpolation']
    for lab,row in zip(labels,points): ax.plot(sigma,sigma*float(row.test_sharpe),label=f"{lab}: C={row.mean_complexity:.0f}, SR={row.test_sharpe:.2f}")
    ax.set_xlabel('Annualized volatility'); ax.set_ylabel('Annualized expected excess return'); ax.set_title('Learned capital-allocation lines from one representation'); ax.legend(frameon=False); savefig(fig,'fig_emp_05_learned_frontiers')

    meta=dict(data='Official public U.S. Jensen-Kelly-Pedersen characteristic-managed portfolios',universe='153 capped-value-weighted characteristic portfolios',state='Lagged 12-month mean and volatility of seven literature groups (14 coordinates)',calibration_end='1994-12-31',validation='1995-2004',test='2005-2025',representation='Matérn-3/2 only',ell_grid=ELL_GRID.tolist(),C_grid=C_GRID.tolist(),selected=selected,test_loss_min_C=test_min,history=history.to_dict(orient='records'))
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    main()
