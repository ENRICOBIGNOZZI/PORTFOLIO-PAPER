def design_levels(design: str) -> list[float]:
    if design == "SNR":
        return [x * x / 12.0 for x in SNR_LEVELS]
    if design == "N":
        return [float(x) for x in N_LEVELS]
    return [float(x) for x in D_LEVELS]


def design_title(design: str) -> str:
    return {"SNR": "Signal-to-noise", "N": "Number of risky assets", "d": "State dimension"}[design]


def line_label(design: str, level: float, level_label: str) -> str:
    if design == "SNR":
        return f"{level_label}, SNR={level:.3f}"
    return level_label


def save_model_matrix(summ: pd.DataFrame, model: str, rates: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), constrained_layout=True)
    designs = ("SNR", "N", "d")
    for col, design in enumerate(designs):
        panel = summ[(summ["model"].eq(model)) & (summ["design"].eq(design))]
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        for level in sorted(panel["level"].unique()):
            cell = panel[np.isclose(panel["level"], level)].sort_values("T")
            label = line_label(design, level, str(cell["level_label"].iloc[0]))
            line, = ax_top.plot(cell["T"], cell["sharpe_median"], marker="o", linewidth=1.7, label=label)
            ax_top.fill_between(cell["T"], cell["sharpe_p10"], cell["sharpe_p90"], alpha=0.11)
            if design == "SNR":
                ax_top.axhline(float(cell["oracle_sr_annual"].iloc[0]), color=line.get_color(), linestyle="--", linewidth=0.9, alpha=0.8)
            ax_bottom.plot(cell["T"], cell["shortfall_median"], marker="o", linewidth=1.7, label=label)
            if model == "Sobolev" and design == "d":
                theory = float(cell["theory_rate"].iloc[0])
                anchor_T = float(cell["T"].iloc[-1])
                anchor_y = float(cell["shortfall_median"].iloc[-1])
                guide = anchor_y * (cell["T"].to_numpy(dtype=float) / anchor_T) ** (-theory)
                ax_bottom.plot(cell["T"], guide, linestyle="--", linewidth=0.9, color=line.get_color(), alpha=0.75)
        if design != "SNR":
            ax_top.axhline(1.50, linestyle="--", linewidth=1.1, label=r"Population $SR^\star=1.50$")
        if not (model == "Sobolev" and design == "d"):
            theory = 1.0 if model == "Linear" else 2.0 / 3.0
            ref = panel[np.isclose(panel["level"], sorted(panel["level"].unique())[0])].sort_values("T")
            anchor_T = float(ref["T"].iloc[-1])
            anchor_y = float(ref["shortfall_median"].iloc[-1])
            guide = anchor_y * (ref["T"].to_numpy(dtype=float) / anchor_T) ** (-theory)
            ax_bottom.plot(ref["T"], guide, linestyle="--", linewidth=1.15, label=fr"Reference $T^{{-{theory:.3g}}}$")
        ax_top.set_xscale("log")
        ax_bottom.set_xscale("log")
        ax_bottom.set_yscale("log")
        ax_top.set_title(design_title(design))
        ax_top.grid(True, which="both", alpha=0.20)
        ax_bottom.grid(True, which="both", alpha=0.20)
        ax_bottom.set_xlabel("Sample size $T$")
        if col == 0:
            ax_top.set_ylabel("Annualized population Sharpe")
            ax_bottom.set_ylabel(r"$1-SR(\widehat w_T)^2/(SR^\star)^2$")
        ax_top.legend(frameon=False, fontsize=7.6, loc="lower right")
        ax_bottom.legend(frameon=False, fontsize=7.6, loc="upper right")
    fig.suptitle(
        "Correctly specified linear portfolio policy" if model == "Linear" else "Correctly specified Sobolev portfolio policy",
        fontsize=14,
    )
    stem = "mc_linear_symmetric_matrix" if model == "Linear" else "mc_sobolev_symmetric_matrix"
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=260)
    plt.close(fig)

