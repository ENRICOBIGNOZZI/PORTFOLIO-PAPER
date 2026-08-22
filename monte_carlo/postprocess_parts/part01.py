def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["model", "design", "level", "level_label", "T"]).agg(
        sharpe_median=("sharpe_annual", "median"),
        sharpe_p10=("sharpe_annual", lambda x: np.quantile(x, 0.10)),
        sharpe_p90=("sharpe_annual", lambda x: np.quantile(x, 0.90)),
        recovery_median=("sharpe2_recovery", "median"),
        recovery_p10=("sharpe2_recovery", lambda x: np.quantile(x, 0.10)),
        recovery_p90=("sharpe2_recovery", lambda x: np.quantile(x, 0.90)),
        shortfall_median=("relative_shortfall", "median"),
        shortfall_mean=("relative_shortfall", "mean"),
        oracle_sr_annual=("oracle_sr_annual", "first"),
        portfolio_snr=("portfolio_snr", "first"),
        theory_rate=("theory_rate", "first"),
        replications=("rep", "nunique"),
    ).reset_index()


def bootstrap_rate(cell: pd.DataFrame, n_tail: int, n_boot: int, seed: int) -> dict[str, float]:
    ts = np.array(sorted(cell["T"].unique()))[-n_tail:]
    values = {T: cell.loc[cell["T"].eq(T), "relative_shortfall"].to_numpy(dtype=float) for T in ts}

    def exponent(medians: np.ndarray) -> float:
        return float(-np.polyfit(np.log(ts), np.log(medians), 1)[0])

    point = exponent(np.array([np.median(values[T]) for T in ts]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        medians = [np.median(rng.choice(values[T], size=values[T].size, replace=True)) for T in ts]
        boots[b] = exponent(np.asarray(medians))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "tail_points": n_tail,
        "first_T": int(ts[0]),
        "last_T": int(ts[-1]),
        "empirical_rate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }

