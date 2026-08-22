#!/usr/bin/env python3
"""Execution, reporting, and plotting for the finance Monte Carlo."""
from finance_mc_core import *  # noqa: F401,F403


def pilot_ridge_scale(T: int, reps: int, candidates: Iterable[float], seed: int,
                      n_jobs: int) -> tuple[float, pd.DataFrame]:
    rows: list[dict] = []
    for c in candidates:
        J_nonlin = nonlinear_feature_count(T, c)
        modes = fourier_modes(D, (J_nonlin - 1) // 2)
        eval_features = {
            "Linear": legendre_features(Z_EVAL, 1),
            "Quadratic": legendre_features(Z_EVAL, 2),
            "Sobolev": flexible_features(Z_EVAL, modes, S),
        }
        tasks = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(exp2_task)(T, rep, seed + int(c * 1_000_000), c, modes, eval_features)
            for rep in range(reps)
        )
        flat = [row for block in tasks for row in block if row["method"] == "Sobolev"]
        for row in flat:
            row = dict(row)
            row["ridge_scale"] = c
            rows.append(row)
    df = pd.DataFrame(rows)
    summary = df.groupby("ridge_scale").agg(
        median_shortfall=("sharpe2_shortfall", "median"),
        median_sharpe=("sharpe_annual", "median"),
    ).reset_index()
    best = float(summary.sort_values(["median_shortfall", "ridge_scale"]).iloc[0]["ridge_scale"])
    return best, summary


def run_experiment_1(T_grid: list[int], reps: int, seed: int, n_jobs: int) -> pd.DataFrame:
    eval_features = {
        "Linear": legendre_features(Z_EVAL, 1),
        "Quadratic": legendre_features(Z_EVAL, 2),
        "Cubic": legendre_features(Z_EVAL, 3),
    }
    jobs = [(T, rep) for T in T_grid for rep in range(reps)]
    blocks = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(exp1_task)(T, rep, seed, eval_features) for T, rep in jobs
    )
    return pd.DataFrame([row for block in blocks for row in block])


def run_experiment_2(T_grid: list[int], reps: int, seed: int, ridge_scale: float,
                     n_jobs: int) -> pd.DataFrame:
    rows: list[dict] = []
    for T in T_grid:
        J_nonlin = nonlinear_feature_count(T, ridge_scale)
        modes = fourier_modes(D, (J_nonlin - 1) // 2)
        eval_features = {
            "Linear": legendre_features(Z_EVAL, 1),
            "Quadratic": legendre_features(Z_EVAL, 2),
            "Sobolev": flexible_features(Z_EVAL, modes, S),
        }
        blocks = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(exp2_task)(T, rep, seed, ridge_scale, modes, eval_features)
            for rep in range(reps)
        )
        rows.extend(row for block in blocks for row in block)
    return pd.DataFrame(rows)


def run_rate_extension(T_grid: list[int], reps: int, seed: int, ridge_scale: float,
                       n_jobs: int) -> pd.DataFrame:
    rows: list[dict] = []
    for T in T_grid:
        J_nonlin = nonlinear_feature_count(T, ridge_scale)
        modes = fourier_modes(D, (J_nonlin - 1) // 2)
        eval_sobolev = flexible_features(Z_EVAL, modes, S)
        blocks = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(exp2_sobolev_only_task)(T, rep, seed, ridge_scale, modes, eval_sobolev)
            for rep in range(reps)
        )
        rows.extend(blocks)
    return pd.DataFrame(rows)


def curve_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["experiment", "method", "T"]).agg(
        sharpe_median=("sharpe_annual", "median"),
        sharpe_p10=("sharpe_annual", lambda x: np.quantile(x, 0.10)),
        sharpe_p90=("sharpe_annual", lambda x: np.quantile(x, 0.90)),
        recovery_median=("sharpe2_recovery", "median"),
        recovery_p10=("sharpe2_recovery", lambda x: np.quantile(x, 0.10)),
        recovery_p90=("sharpe2_recovery", lambda x: np.quantile(x, 0.90)),
        shortfall_median=("sharpe2_shortfall", "median"),
        shortfall_mean=("sharpe2_shortfall", "mean"),
        n_features=("n_features", "median"),
        replications=("rep", "nunique"),
    ).reset_index()


def slope_from_summary(summary: pd.DataFrame, n_tail: int) -> float:
    cell = summary.sort_values("T").tail(n_tail)
    slope = np.polyfit(np.log(cell["T"]), np.log(cell["shortfall_median"]), 1)[0]
    return float(-slope)


def bootstrap_rate(exp2: pd.DataFrame, n_tail: int = 6, n_boot: int = 2000,
                   seed: int = 20260818) -> dict[str, float]:
    cell = exp2[exp2["method"] == "Sobolev"].copy()
    Ts = np.array(sorted(cell["T"].unique()))[-n_tail:]
    values = {T: cell.loc[cell["T"] == T, "sharpe2_shortfall"].to_numpy() for T in Ts}

    def exponent(medians: np.ndarray) -> float:
        return float(-np.polyfit(np.log(Ts), np.log(medians), 1)[0])

    point = exponent(np.array([np.median(values[T]) for T in Ts]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        med = []
        for T in Ts:
            v = values[T]
            med.append(np.median(rng.choice(v, size=v.size, replace=True)))
        boots[b] = exponent(np.asarray(med))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"tail_points": n_tail, "empirical_exponent": point,
            "ci_low": float(lo), "ci_high": float(hi),
            "theory_exponent": THEORY_RATE}


def rate_sensitivity(exp2: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([bootstrap_rate(exp2, n_tail=n, n_boot=1200, seed=9000+n)
                         for n in (4, 5, 6)])


def save_line_with_band(summary: pd.DataFrame, metric: str, low: str, high: str,
                        ylabel: str, title: str, path_stem: str,
                        oracle_line: float | None = None,
                        percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    order = [m for m in ("Linear", "Quadratic", "Cubic", "Sobolev")
             if m in summary["method"].unique()]
    for method in order:
        cell = summary[summary["method"] == method].sort_values("T")
        y = cell[metric].to_numpy(); lo = cell[low].to_numpy(); hi = cell[high].to_numpy()
        if percent:
            y, lo, hi = 100*y, 100*lo, 100*hi
        ax.plot(cell["T"], y, marker="o", linewidth=1.8, label=method)
        ax.fill_between(cell["T"], lo, hi, alpha=0.14)
    if oracle_line is not None:
        ax.axhline(oracle_line, linestyle="--", linewidth=1.3, label="Population optimum")
    ax.set_xscale("log"); ax.set_xlabel("Sample size $T$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGURES / f"{path_stem}.pdf"); fig.savefig(FIGURES / f"{path_stem}.png", dpi=240); plt.close(fig)


def make_rate_figure(exp2_summary: pd.DataFrame, rate_main: dict[str, float]) -> None:
    cell = exp2_summary[exp2_summary["method"] == "Sobolev"].sort_values("T")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(cell["T"], cell["shortfall_median"], marker="o", linewidth=1.9,
            label="Median squared-Sharpe shortfall")
    anchor_T=float(cell["T"].iloc[-1]); anchor_y=float(cell["shortfall_median"].iloc[-1])
    guide=anchor_y*(cell["T"].to_numpy()/anchor_T)**(-THEORY_RATE)
    ax.plot(cell["T"],guide,linestyle="--",linewidth=1.4,label=fr"Theoretical slope $T^{{-{THEORY_RATE:.3f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Sample size $T$")
    ax.set_ylabel(r"$(SR^{\star})^2-SR(\widehat w)^2$")
    ax.set_title("Nonlinear economy: Sharpe learning rate\n"+f"empirical exponent {rate_main['empirical_exponent']:.3f} [{rate_main['ci_low']:.3f}, {rate_main['ci_high']:.3f}]")
    ax.grid(True,which="both",alpha=0.22); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(FIGURES/"mc_nonlinear_rate.pdf"); fig.savefig(FIGURES/"mc_nonlinear_rate.png",dpi=240); plt.close(fig)


def representative_policy_figure(exp2: pd.DataFrame, ridge_scale: float, seed: int,
                                 T_rep: int = 2000) -> int:
    available=np.array(sorted(exp2["T"].unique()))
    if T_rep not in available: T_rep=int(available[np.argmin(np.abs(available-T_rep))])
    cell=exp2[(exp2["T"]==T_rep)&(exp2["method"]=="Sobolev")].copy(); target=float(cell["sharpe_annual"].median())
    rep=int(cell.iloc[(cell["sharpe_annual"]-target).abs().argmin()]["rep"])
    rng=np.random.default_rng(seed+300_000_000+100_000*T_rep+rep); z=simulate_state(T_rep,D,ECON.rho,rng); R=sample_returns(lambda_nonlinear(z),rng)
    J_nonlin=nonlinear_feature_count(T_rep,ridge_scale); modes=fourier_modes(D,(J_nonlin-1)//2); ridge=ridge_scale*T_rep**(-THEORY_LAMBDA_EXP)
    coef_lin=fit_direct_policy(legendre_features(z,1),R,0.25/T_rep); coef_sob=fit_direct_policy(flexible_features(z,modes,S),R,ridge)
    grid=np.linspace(0.02,0.98,90); z1,z2=np.meshgrid(grid,grid); z_slice=np.column_stack([z1.ravel(),z2.ravel(),np.full(z1.size,0.50)])
    w_true=true_policy(lambda_nonlinear(z_slice))[:,0].reshape(z1.shape); w_lin=evaluate_policy(legendre_features(z_slice,1),coef_lin)[:,0].reshape(z1.shape); w_sob=evaluate_policy(flexible_features(z_slice,modes,S),coef_sob)[:,0].reshape(z1.shape)
    values=[w_true,w_lin,w_sob]; titles=["Population policy","Estimated linear policy","Estimated Sobolev policy"]; names=["mc_policy_true","mc_policy_linear","mc_policy_sobolev"]
    vmin=min(float(v.min()) for v in values); vmax=max(float(v.max()) for v in values)
    for value,title,name in zip(values,titles,names):
        fig,ax=plt.subplots(figsize=(6.2,5.0)); image=ax.imshow(value,origin="lower",extent=[0.02,0.98,0.02,0.98],aspect="auto",vmin=vmin,vmax=vmax)
        ax.set_xlabel("State $Z_{1t}$"); ax.set_ylabel("State $Z_{2t}$"); ax.set_title(title+r" for asset 1 at $Z_{3t}=0.5$")
        fig.colorbar(image,ax=ax,label="Portfolio exposure"); fig.tight_layout(); fig.savefig(FIGURES/f"{name}.pdf"); fig.savefig(FIGURES/f"{name}.png",dpi=240); plt.close(fig)
    return rep


def make_figures(exp1: pd.DataFrame, exp2: pd.DataFrame, rate_df: pd.DataFrame,
                 rate_main: dict[str, float], ridge_scale: float, seed: int) -> int:
    FIGURES.mkdir(parents=True,exist_ok=True); s1=curve_summary(exp1); s2=curve_summary(exp2)
    save_line_with_band(s1,"sharpe_median","sharpe_p10","sharpe_p90","Annualized population Sharpe ratio","Linear factor premia: correctly specified policy","mc_linear_sharpe",oracle_line=TARGET_ANNUAL_SR)
    save_line_with_band(s1,"recovery_median","recovery_p10","recovery_p90","Squared Sharpe recovered (percent)","Linear factor premia: fraction of the opportunity recovered","mc_linear_recovery",oracle_line=100.0,percent=True)
    save_line_with_band(s2,"sharpe_median","sharpe_p10","sharpe_p90","Annualized population Sharpe ratio","Nonlinear factor premia: flexible portfolio learning","mc_nonlinear_sharpe",oracle_line=TARGET_ANNUAL_SR)
    save_line_with_band(s2,"recovery_median","recovery_p10","recovery_p90","Squared Sharpe recovered (percent)","Nonlinear factor premia: fraction of the opportunity recovered","mc_nonlinear_recovery",oracle_line=100.0,percent=True)
    make_rate_figure(curve_summary(rate_df),rate_main); return representative_policy_figure(exp2,ridge_scale,seed)


def final_T_table(exp1: pd.DataFrame, exp2: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for label,df in (("Linear economy",exp1),("Nonlinear economy",exp2)):
        T=int(df["T"].max()); cell=df[df["T"]==T]
        for method,x in cell.groupby("method"):
            rows.append({"economy":label,"T":T,"method":method,"median_sharpe":float(x["sharpe_annual"].median()),"p10_sharpe":float(x["sharpe_annual"].quantile(.10)),"p90_sharpe":float(x["sharpe_annual"].quantile(.90)),"median_recovery_pct":100.0*float(x["sharpe2_recovery"].median())})
    return pd.DataFrame(rows)


def write_tables(final_table: pd.DataFrame, rate_table: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True,exist_ok=True)
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Economy & Policy & $T$ & Median SR & 10--90\% range & SR$^2$ recovered \\",r"\midrule"]
    for _,x in final_table.iterrows():
        interval=f"[{x.p10_sharpe:.2f}, {x.p90_sharpe:.2f}]"; lines.append(f"{x.economy} & {x.method} & {int(x['T'])} & {x.median_sharpe:.2f} & {interval} & {x.median_recovery_pct:.1f}\\% "+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}"]; (TABLES/"mc_sharpe_summary.tex").write_text("\n".join(lines)+"\n")
    lines=[r"\begin{tabular}{rrrr}",r"\toprule",r"Tail points & Theory & Empirical & 95\% bootstrap interval \\",r"\midrule"]
    for _,x in rate_table.iterrows(): lines.append(f"{int(x.tail_points)} & {x.theory_exponent:.3f} & {x.empirical_exponent:.3f} & [{x.ci_low:.3f}, {x.ci_high:.3f}] "+r"\\")
    lines += [r"\bottomrule",r"\end{tabular}"]; (TABLES/"mc_rate_summary.tex").write_text("\n".join(lines)+"\n")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--reps",type=int,default=200); parser.add_argument("--rate-reps",type=int,default=120); parser.add_argument("--seed",type=int,default=20260818); parser.add_argument("--jobs",type=int,default=-1); parser.add_argument("--quick",action="store_true"); parser.add_argument("--ridge-scale",type=float,default=None); args=parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True); TABLES.mkdir(parents=True,exist_ok=True)
    if args.quick:
        T_grid=[120,300,700]; rate_T_grid=[1200,2500]; reps=min(args.reps,20); rate_reps=min(args.rate_reps,12); pilot_reps=10
    else:
        T_grid=[120,200,320,500,800,1250,2000]; rate_T_grid=[3200,5000,8000,12000,18000]; reps=args.reps; rate_reps=args.rate_reps; pilot_reps=40
    checks=oracle_checks()
    if checks["max_norm_lambda_linear"]>=0.95 or checks["max_norm_lambda_nonlinear"]>=0.95: raise RuntimeError(f"Invalid signal calibration: {checks}")
    if args.ridge_scale is None:
        ridge_scale,pilot=pilot_ridge_scale(T=600 if not args.quick else 300,reps=pilot_reps,candidates=[0.5,1.0,2.0,4.0,8.0],seed=args.seed+77_000_000,n_jobs=args.jobs)
    else:
        ridge_scale=float(args.ridge_scale); pilot=pd.DataFrame({"ridge_scale":[ridge_scale]})
    exp1=run_experiment_1(T_grid,reps,args.seed,args.jobs); exp2=run_experiment_2(T_grid,reps,args.seed+123_456,ridge_scale,args.jobs); rate_extension=run_rate_extension(rate_T_grid,rate_reps,args.seed+987_654,ridge_scale,args.jobs)
    rate_df=pd.concat([exp2[exp2["method"]=="Sobolev"].copy(),rate_extension],ignore_index=True); summary=pd.concat([curve_summary(exp1),curve_summary(exp2)],ignore_index=True); rate_table=rate_sensitivity(rate_df); rate_main=rate_table.loc[rate_table["tail_points"]==5].iloc[0].to_dict(); final_table=final_T_table(exp1,exp2)
    exp1.to_csv(RESULTS/"mc_linear_raw.csv",index=False); exp2.to_csv(RESULTS/"mc_nonlinear_raw.csv",index=False); rate_df.to_csv(RESULTS/"mc_rate_raw.csv",index=False); summary.to_csv(RESULTS/"mc_curve_summary.csv",index=False); rate_table.to_csv(RESULTS/"mc_rate_summary.csv",index=False); final_table.to_csv(RESULTS/"mc_final_T_summary.csv",index=False); pilot.to_csv(RESULTS/"mc_ridge_pilot.csv",index=False)
    representative_rep=make_figures(exp1,exp2,rate_df,rate_main,ridge_scale,args.seed+123_456); write_tables(final_table,rate_table)
    metadata={"seed":args.seed,"replications_per_T":reps,"rate_replications_per_T":rate_reps,"T_grid":T_grid,"rate_T_grid":rate_T_grid,"state_dimension":D,"assets":ECON.N,"factors":ECON.K,"state_persistence":ECON.rho,"target_annual_sharpe":TARGET_ANNUAL_SR,"target_m":TARGET_M,"sobolev_s":S,"source_r":R_SOURCE,"spectral_b":B_SPECTRAL,"theory_rate":THEORY_RATE,"theory_lambda_exponent":THEORY_LAMBDA_EXP,"selected_ridge_scale":ridge_scale,"representative_replication":representative_rep,"oracle_checks":checks}; (RESULTS/"mc_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
if __name__=="__main__": main()
