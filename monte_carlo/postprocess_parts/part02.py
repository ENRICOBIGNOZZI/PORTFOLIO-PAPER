def rate_table(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, ((model, design, level), cell) in enumerate(df.groupby(["model", "design", "level"], sort=True)):
        stats = bootstrap_rate(cell, n_tail=6, n_boot=n_boot, seed=seed + idx * 100)
        first = cell.iloc[0]
        rows.append({
            "model": model,
            "design": design,
            "level": level,
            "level_label": first["level_label"],
            "oracle_sr_annual": first["oracle_sr_annual"],
            "portfolio_snr": first["portfolio_snr"],
            "theory_rate": first["theory_rate"],
            **stats,
        })
    return pd.DataFrame(rows).sort_values(["model", "design", "level"]).reset_index(drop=True)


def history_required(summ: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, cell in summ.groupby(["model", "design", "level", "level_label"], sort=True):
        model, design, level, level_label = keys
        cell = cell.sort_values("T")
        T = cell["T"].to_numpy(dtype=float)
        recovery = cell["recovery_median"].to_numpy(dtype=float)
        monotone = np.maximum.accumulate(recovery)
        for threshold in THRESHOLDS:
            left_censored = False
            if monotone[0] >= threshold:
                required = T[0]
                reached = True
                left_censored = True
            elif monotone[-1] < threshold:
                required = math.nan
                reached = False
            else:
                j = int(np.where(monotone >= threshold)[0][0])
                fraction = (threshold - monotone[j - 1]) / max(monotone[j] - monotone[j - 1], 1e-12)
                required = float(np.exp(np.log(T[j - 1]) + fraction * (np.log(T[j]) - np.log(T[j - 1]))))
                reached = True
            rows.append({
                "model": model,
                "design": design,
                "level": level,
                "level_label": level_label,
                "threshold": threshold,
                "T_required": required,
                "reached": reached,
                "left_censored": left_censored,
                "largest_T": int(T[-1]),
            })
    return pd.DataFrame(rows)

