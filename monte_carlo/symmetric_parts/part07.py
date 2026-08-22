def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", choices=["all", "linear_snr", "linear_N", "dimension"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.quick:
        snr_T = (800, 3200)
        n_T = (800, 3200)
        d_T = (800, 3200)
        snr_reps = {T: 3 for T in snr_T}
        n_reps = {T: 3 for T in n_T}
        d_reps = {T: 3 for T in d_T}
    else:
        snr_T = SNR_T
        n_T = N_T
        d_T = D_T
        snr_reps = {T: 50 for T in snr_T}
        n_reps = {T: (48 if T <= 18000 else 24) for T in n_T}
        d_reps = {T: (40 if T <= 12000 else (30 if T == 18000 else (24 if T == 28000 else (16 if T == 42000 else 12)))) for T in d_T}

    linear_snr_path = RESULTS / "mc_symmetric_linear_snr_raw.csv"
    linear_n_path = RESULTS / "mc_symmetric_linear_N_raw.csv"
    dimension_path = RESULTS / "mc_symmetric_dimension_raw.csv"

    if args.only in ("all", "linear_snr"):
        linear_snr = run_tasks_incremental(
            linear_snr_task, snr_T, snr_reps, args.seed + 10_000, args.jobs,
            linear_snr_path, ["T", "rep", "oracle_sr_annual"], args.resume,
        )
    else:
        linear_snr = pd.read_csv(linear_snr_path) if linear_snr_path.exists() else pd.DataFrame()
    if args.only in ("all", "linear_N"):
        linear_n = run_tasks_incremental(
            linear_n_task, n_T, n_reps, args.seed + 20_000, args.jobs,
            linear_n_path, ["T", "rep", "N"], args.resume,
        )
    else:
        linear_n = pd.read_csv(linear_n_path) if linear_n_path.exists() else pd.DataFrame()
    if args.only in ("all", "dimension"):
        dimension = run_tasks_incremental(
            state_dimension_task, d_T, d_reps, args.seed + 30_000, args.jobs,
            dimension_path, ["T", "rep", "model", "d"], args.resume,
        )
    else:
        dimension = pd.read_csv(dimension_path) if dimension_path.exists() else pd.DataFrame()

    checks = {
        "linear_dimension": {str(d): linear_dimension_oracle_check(d) for d in D_LEVELS},
        "sobolev_dimension": {},
        "replications": {
            "linear_snr": {str(T): snr_reps[T] for T in snr_T},
            "linear_N": {str(T): n_reps[T] for T in n_T},
            "state_dimension": {str(T): d_reps[T] for T in d_T},
        },
        "T_grids": {"snr": list(snr_T), "N": list(n_T), "d": list(d_T)},
        "ridge_scales": {"linear": RIDGE_LINEAR_SCALE, "sobolev": RIDGE_SOBOLEV_SCALE},
    }
    for d, design in SOBOLEV_DESIGNS.items():
        oracle_w = mc.true_policy(design.lambda_eval, mc.ECON)
        metrics = population_metrics(oracle_w, design.lambda_eval, mc.ECON, BASE_SR)
        checks["sobolev_dimension"][str(d)] = {
            "quadrature_oracle_sharpe": metrics["sharpe_annual"],
            "max_norm_lambda": float(np.linalg.norm(design.lambda_eval, axis=1).max()),
            "theory_rate": design.theory_rate,
            "ridge_exponent": design.ridge_exponent,
        }
    (RESULTS / "mc_symmetric_design_metadata.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )

    if not linear_snr.empty:
        print("LINEAR SNR FINAL MEDIANS")
        final_t = int(linear_snr["T"].max())
        print(linear_snr[linear_snr["T"] == final_t].groupby("oracle_sr_annual")["sharpe_annual"].median())
    if not linear_n.empty:
        print("\nLINEAR N FINAL MEDIANS")
        final_t = int(linear_n["T"].max())
        print(linear_n[linear_n["T"] == final_t].groupby("N")["sharpe_annual"].median())
    if not dimension.empty:
        print("\nDIMENSION FINAL MEDIANS")
        final_t = int(dimension["T"].max())
        print(dimension[dimension["T"] == final_t].groupby(["model", "d"])["sharpe_annual"].median())
    print("\nORACLE CHECKS")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
