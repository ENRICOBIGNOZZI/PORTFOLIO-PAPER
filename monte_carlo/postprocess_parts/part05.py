def format_level(design: str, level: float, oracle_sr: float) -> str:
    if design == "SNR":
        return f"$SR^\\star={oracle_sr:.1f}$"
    if design == "N":
        return f"$N={int(level)}$"
    return f"$d={int(level)}$"


def write_main_table(rates: pd.DataFrame, history: pd.DataFrame) -> None:
    rows = []
    for _, rate in rates.iterrows():
        hist = history[
            history["model"].eq(rate.model)
            & history["design"].eq(rate.design)
            & np.isclose(history["level"], rate.level)
        ].set_index("threshold")
        def ft(th: float) -> str:
            value = hist.loc[th, "T_required"]
            if not np.isfinite(value):
                return f"$>{int(hist.loc[th, 'largest_T']):,}$"
            if bool(hist.loc[th, "left_censored"]):
                return f"$\\leq {value:,.0f}$"
            return f"{value:,.0f}"
        rows.append({
            "model": rate.model,
            "design": rate.design,
            "level_text": format_level(rate.design, rate.level, rate.oracle_sr_annual),
            "theory": rate.theory_rate,
            "empirical": rate.empirical_rate,
            "ci_low": rate.ci_low,
            "ci_high": rate.ci_high,
            "T75": ft(0.75),
            "T90": ft(0.90),
        })
    table = pd.DataFrame(rows)
    order_design = {"SNR": 0, "N": 1, "d": 2}
    order_model = {"Linear": 0, "Sobolev": 1}
    table["model_order"] = table["model"].map(order_model)
    table["design_order"] = table["design"].map(order_design)
    table["level_order"] = rates.sort_values(["model", "design", "level"])["level"].to_numpy()
    table = table.sort_values(["model_order", "design_order", "level_order"]).drop(columns=["model_order", "design_order", "level_order"])
    lines = [
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Model & Margin & Cell & Theory & Empirical [95\% CI] & $T_{90\%}$ \\",
        r"\midrule",
    ]
    last_model = None
    for _, x in table.iterrows():
        model_text = x.model if x.model != last_model else ""
        lines.append(
            f"{model_text} & {x.design} & {x.level_text} & {x.theory:.3f} & "
            f"{x.empirical:.3f} [{x.ci_low:.3f},{x.ci_high:.3f}] & {x.T90} " + r"\\"
        )
        last_model = x.model
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mc_symmetric_main_summary.tex").write_text("\n".join(lines) + "\n")
    table.to_csv(RESULTS / "mc_symmetric_main_summary.csv", index=False)

