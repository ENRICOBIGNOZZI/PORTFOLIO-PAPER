#!/usr/bin/env python3
"""Summaries, figures, tables, and audit checks for N and persistence experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import finance_monte_carlo as mc
import finance_monte_carlo_expanded as ex
import comparative_statics_mc as cs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

plt.rcParams.update({"font.size":10.5,"axes.titlesize":12,"axes.labelsize":10.5,"legend.fontsize":9.0,"figure.dpi":130,"savefig.bbox":"tight"})

def history_required(summary:pd.DataFrame,group:str,thresholds:tuple[float,...])->pd.DataFrame:
    rows=[]
    for value,cell in summary.groupby(group):
        cell=cell.sort_values("T"); t=cell["T"].to_numpy(float); recovery=cell["recovery_median"].to_numpy(float)
        for threshold in thresholds:
            reached=bool(recovery[-1]>=threshold)
            if recovery[0]>=threshold: required=float(t[0])
            elif not reached: required=math.nan
            else:
                idx=int(np.where(recovery>=threshold)[0][0]); log_t0,log_t1=np.log(t[idx-1]),np.log(t[idx]); y0,y1=recovery[idx-1],recovery[idx]; fraction=(threshold-y0)/(y1-y0); required=float(np.exp(log_t0+fraction*(log_t1-log_t0)))
            rows.append({group:value,"threshold":threshold,"T_required":required,"reached":reached,"largest_T":int(t[-1])})
    return pd.DataFrame(rows)

def paired_difference(df:pd.DataFrame,design_col:str,reference:float,alternative:float,t:int,n_boot:int,seed:int)->dict:
    cell=df[df["T"].eq(t)].pivot(index="rep",columns=design_col,values="sharpe_annual").dropna(); differences=(cell[alternative]-cell[reference]).to_numpy(float); rng=np.random.default_rng(seed); boot=np.empty(n_boot)
    for b in range(n_boot): boot[b]=rng.choice(differences,size=differences.size,replace=True).mean()
    lo,hi=np.quantile(boot,[.025,.975]); return {"design":design_col,"reference":reference,"alternative":alternative,"T":t,"replications":int(differences.size),"median_difference":float(np.median(differences)),"mean_difference":float(differences.mean()),"ci_low":float(lo),"ci_high":float(hi)}

def rate_rows(df:pd.DataFrame,group:str,values:list[float],n_tail:int,n_boot:int,seed:int):
    rates=[]; rolling=[]
    for idx,value in enumerate(values):
        cell=df[np.isclose(df[group],value)].copy(); row=ex.bootstrap_exponent(cell,mc.THEORY_RATE,n_tail=n_tail,n_boot=n_boot,seed=seed+100*idx); row[group]=value; rates.append(row); local=ex.rolling_exponents(cell,mc.THEORY_RATE,window=4,n_boot=min(800,n_boot),seed=seed+10_000+100*idx); local[group]=value; rolling.append(local)
    return pd.DataFrame(rates),pd.concat(rolling,ignore_index=True)

def save_curve(summary:pd.DataFrame,group:str,ylabel:str,metric:str,low:str,high:str,title:str,filename:str,oracle:bool=False,ylog:bool=False)->None:
    fig,ax=plt.subplots(figsize=(7.5,4.9))
    for value,cell in summary.groupby(group,sort=True):
        cell=cell.sort_values("T"); ax.plot(cell["T"],cell[metric],marker="o",linewidth=1.8,label=f"{group}={value:g}"); ax.fill_between(cell["T"],cell[low],cell[high],alpha=.13)
    if oracle: ax.axhline(mc.TARGET_ANNUAL_SR,linestyle="--",linewidth=1.25,label=r"Population $SR^\star=1.50$")
    ax.set_xscale("log");
    if ylog: ax.set_yscale("log")
    ax.set_xlabel("Sample size $T$"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True,which="both",alpha=.22); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(FIGURES/f"{filename}.pdf"); fig.savefig(FIGURES/f"{filename}.png",dpi=260); plt.close(fig)

def save_dimension_local(rolling:pd.DataFrame)->None:
    fig,ax=plt.subplots(figsize=(7.5,4.9))
    for n,cell in rolling.groupby("N",sort=True):
        cell=cell.sort_values("endpoint_T"); ax.plot(cell["endpoint_T"],cell["empirical_exponent"],marker="o",linewidth=1.8,label=f"N={int(n)}"); ax.fill_between(cell["endpoint_T"],cell["ci_low"],cell["ci_high"],alpha=.12)
    ax.axhline(mc.THEORY_RATE,linestyle="--",linewidth=1.25,label=r"Theory $2/3$"); ax.axhspan(mc.THEORY_RATE-.08,mc.THEORY_RATE+.08,alpha=.06); ax.set_xscale("log"); ax.set_xlabel("Right endpoint of four-cell window"); ax.set_ylabel("Local convergence exponent"); ax.set_title("Asset dimension delays the asymptotic learning regime"); ax.grid(True,which="both",alpha=.22); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(FIGURES/"mc_dimension_local_exponent.pdf"); fig.savefig(FIGURES/"mc_dimension_local_exponent.png",dpi=260); plt.close(fig)

def save_history(history:pd.DataFrame)->None:
    fig,ax=plt.subplots(figsize=(7.1,4.8)); max_t=int(history["largest_T"].max())
    for threshold,cell in history.groupby("threshold"):
        cell=cell.sort_values("N"); y=cell["T_required"].fillna(max_t*1.20).to_numpy(float); ax.plot(cell["N"],y,marker="o",linewidth=1.8,label=f"{100*threshold:.0f}% recovery")
        for n,yy,reached in zip(cell["N"],y,cell["reached"]):
            if not bool(reached): ax.annotate(f">{max_t:,}",(n,yy),xytext=(0,5),textcoords="offset points",ha="center",fontsize=8.5)
    ax.set_yscale("log"); ax.set_xlabel("Number of risky assets $N$"); ax.set_ylabel("History required, $T$"); ax.set_title("Output dimension changes the amount of history required"); ax.grid(True,which="both",alpha=.22); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/"mc_dimension_history_required.pdf"); fig.savefig(FIGURES/"mc_dimension_history_required.png",dpi=260); plt.close(fig)

def save_rate_errorbars(rates:pd.DataFrame,group:str,title:str,filename:str)->None:
    rates=rates.sort_values(group); x=rates[group].to_numpy(float); y=rates["empirical_exponent"].to_numpy(float); yerr=np.vstack([y-rates["ci_low"].to_numpy(float),rates["ci_high"].to_numpy(float)-y]); fig,ax=plt.subplots(figsize=(6.6,4.7)); ax.errorbar(x,y,yerr=yerr,fmt="o",capsize=4,linewidth=1.4); ax.axhline(mc.THEORY_RATE,linestyle="--",linewidth=1.25,label=r"Theory $2/3$"); ax.set_xlabel(r"State persistence $\rho$" if group=="rho" else group); ax.set_ylabel("Tail convergence exponent"); ax.set_title(title); ax.grid(True,alpha=.22); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES/f"{filename}.pdf"); fig.savefig(FIGURES/f"{filename}.png",dpi=260); plt.close(fig)

def write_dimension_table(summary:pd.DataFrame,rates:pd.DataFrame,history:pd.DataFrame)->None:
    lines=[r"\begin{tabular}{ccccccc}",r"\toprule",r"$N$ & $SR_{5{,}000}$ & $SR_{18{,}000}$ & $SR_{96{,}000}$ & $T_{75\%}$ & $T_{90\%}$ & Tail exponent \\",r"\midrule"]
    for n in sorted(summary["N"].unique()):
        cell=summary[summary["N"].eq(n)].set_index("T"); rate=rates[rates["N"].eq(n)].iloc[0]; h=history[history["N"].eq(n)].set_index("threshold")
        def fmt_t(threshold:float)->str:
            value=h.loc[threshold,"T_required"]; return f"{value:,.0f}" if np.isfinite(value) else r"$>96{,}000$"
        lines.append(f"{int(n)} & {cell.loc[5000,'sharpe_median']:.3f} & {cell.loc[18000,'sharpe_median']:.3f} & {cell.loc[96000,'sharpe_median']:.3f} & {fmt_t(.75)} & {fmt_t(.90)} & {rate.empirical_exponent:.3f} [{rate.ci_low:.3f},{rate.ci_high:.3f}] "+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}"]; (TABLES/"mc_dimension_summary.tex").write_text("\n".join(lines)+"\n")

def write_persistence_table(summary:pd.DataFrame,rates:pd.DataFrame)->None:
    lines=[r"\begin{tabular}{ccccc}",r"\toprule",r"$\rho$ & $SR_{800}$ & $SR_{5{,}000}$ & $SR_{64{,}000}$ & Tail exponent \\",r"\midrule"]
    for rho in sorted(summary["rho"].unique()):
        cell=summary[np.isclose(summary["rho"],rho)].set_index("T"); rate=rates[np.isclose(rates["rho"],rho)].iloc[0]; lines.append(f"{rho:.2f} & {cell.loc[800,'sharpe_median']:.3f} & {cell.loc[5000,'sharpe_median']:.3f} & {cell.loc[64000,'sharpe_median']:.3f} & {rate.empirical_exponent:.3f} [{rate.ci_low:.3f},{rate.ci_high:.3f}] "+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}"]; (TABLES/"mc_persistence_summary.tex").write_text("\n".join(lines)+"\n")

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--bootstrap",type=int,default=2000); parser.add_argument("--seed",type=int,default=20260818); args=parser.parse_args(); FIGURES.mkdir(exist_ok=True); TABLES.mkdir(exist_ok=True)
    n_df=pd.read_csv(RESULTS/"mc_asset_dimension_raw.csv"); rho_df=pd.read_csv(RESULTS/"mc_persistence_raw.csv"); n_summary=ex.quantile_summary(n_df,["N","T"]); rho_summary=ex.quantile_summary(rho_df,["rho","T"]); n_rates,n_rolling=rate_rows(n_df,"N",[6,20,50],6,args.bootstrap,args.seed+110_000); rho_rates,rho_rolling=rate_rows(rho_df,"rho",[0.00,0.55,0.85,0.95],6,args.bootstrap,args.seed+120_000); n_history=history_required(n_summary,"N",(.50,.75,.90,.95))
    paired_rows=[]
    for t in (5000,18000,96000):
        for n in (20,50): paired_rows.append(paired_difference(n_df,"N",6,n,t,args.bootstrap,args.seed+130_000+t+n))
    for t in (800,1250,2000,5000): paired_rows.append(paired_difference(rho_df,"rho",0.00,0.95,t,args.bootstrap,args.seed+140_000+t))
    paired=pd.DataFrame(paired_rows); n_summary.to_csv(RESULTS/"mc_asset_dimension_summary.csv",index=False); n_rates.to_csv(RESULTS/"mc_asset_dimension_rate_summary.csv",index=False); n_rolling.to_csv(RESULTS/"mc_asset_dimension_rolling_exponents.csv",index=False); n_history.to_csv(RESULTS/"mc_asset_dimension_history_required.csv",index=False); rho_summary.to_csv(RESULTS/"mc_persistence_summary.csv",index=False); rho_rates.to_csv(RESULTS/"mc_persistence_rate_summary.csv",index=False); rho_rolling.to_csv(RESULTS/"mc_persistence_rolling_exponents.csv",index=False); paired.to_csv(RESULTS/"mc_comparative_statics_paired_differences.csv",index=False)
    save_curve(n_summary,"N","Population annualized Sharpe ratio","sharpe_median","sharpe_p10","sharpe_p90","Same investment opportunity, different number of assets","mc_dimension_sharpe_convergence",oracle=True); save_curve(n_summary,"N","Relative squared-Sharpe shortfall","relative_shortfall_median","relative_shortfall_median","relative_shortfall_median","Asset dimension and the pre-asymptotic region","mc_dimension_relative_shortfall",ylog=True); save_dimension_local(n_rolling); save_history(n_history); save_curve(rho_summary,"rho","Population annualized Sharpe ratio","sharpe_median","sharpe_p10","sharpe_p90","Persistence changes short-sample precision, not the population frontier","mc_persistence_sharpe_convergence",oracle=True); save_curve(rho_summary,"rho","Relative squared-Sharpe shortfall","relative_shortfall_median","relative_shortfall_median","relative_shortfall_median","Geometric persistence leaves the large-sample slope intact","mc_persistence_relative_shortfall",ylog=True); save_rate_errorbars(rho_rates,"rho","Tail rates under different state persistence","mc_persistence_tail_exponents"); write_dimension_table(n_summary,n_rates,n_history); write_persistence_table(rho_summary,rho_rates)
    normalization={}
    for n,econ in cs.ECONOMIES.items():
        d=np.diag(econ.idio_sd**2); information=econ.B.T@np.linalg.solve(d,econ.B); oracle_weights=mc.true_policy(mc.LAMBDA_NONLINEAR_EVAL,econ); oracle=cs.population_metrics_econ(oracle_weights,mc.LAMBDA_NONLINEAR_EVAL,econ); normalization[str(n)]={"max_abs_BDinvB_difference":float(np.max(np.abs(information-cs.S_FACTOR))),"max_abs_factor_opportunity_difference":float(np.max(np.abs(econ.H-mc.ECON.H))),"quadrature_oracle_sharpe":float(oracle["sharpe_annual"])}
    metadata={"asset_dimension_levels":[6,20,50],"asset_dimension_T":sorted(int(x) for x in n_df["T"].unique()),"asset_dimension_replications_by_T":{str(int(t)):int(c["rep"].nunique()) for t,c in n_df.groupby("T")},"persistence_levels":[0.00,0.55,0.85,0.95],"persistence_T":sorted(int(x) for x in rho_df["T"].unique()),"persistence_replications_per_cell":int(rho_df.groupby(["rho","T"])["rep"].nunique().min()),"common_population_annual_sharpe":mc.TARGET_ANNUAL_SR,"common_sobolev_s":mc.S,"common_source_r":mc.R_SOURCE,"common_theoretical_exponent":mc.THEORY_RATE,"asset_economy_normalization_checks":normalization,"iterative_solver_validation":cs.validate_iterative_solver(args.seed+17),"rate_estimation":"median relative squared-Sharpe shortfall; six largest T cells; cell bootstrap","history_interpolation":"linear in recovery against log T"}; (RESULTS/"mc_comparative_statics_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
if __name__=="__main__": main()
