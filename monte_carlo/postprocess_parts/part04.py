def save_history_figure(history: pd.DataFrame, design: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True, sharey=True)
    for ax, model in zip(axes, ("Linear", "Sobolev")):
        cell = history[(history["model"].eq(model)) & (history["design"].eq(design))]
        levels = sorted(cell["level"].unique())
        for threshold in THRESHOLDS:
            curve = cell[np.isclose(cell["threshold"], threshold)].sort_values("level")
            max_t = int(curve["largest_T"].max())
            y = curve["T_required"].to_numpy(dtype=float)
            plotted = np.where(np.isfinite(y), y, max_t * 1.22)
            ax.plot(curve["level"], plotted, marker="o", linewidth=1.8, label=f"{100*threshold:.0f}% recovery")
            for x, yy, reached in zip(curve["level"], plotted, curve["reached"]):
                if not bool(reached):
                    ax.annotate(f">{max_t:,}", (x, yy), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22)
        ax.set_title(model)
        if design == "SNR":
            labels = []
            for level in levels:
                sr = math.sqrt(12.0 * level)
                labels.append(f"{level:.3f}\n($SR^\\star={sr:.1f}$)")
            ax.set_xticks(levels, labels)
            ax.set_xlabel("Portfolio SNR")
        elif design == "N":
            ax.set_xticks(levels, [str(int(x)) for x in levels])
            ax.set_xlabel("Number of risky assets $N$")
        else:
            ax.set_xticks(levels, [str(int(x)) for x in levels])
            ax.set_xlabel("State dimension $d$")
        ax.legend(frameon=False)
    axes[0].set_ylabel("History required, $T$")
    fig.suptitle(f"History required: {design_title(design)}", fontsize=13)
    stem = {"SNR": "mc_history_required_snr_models", "N": "mc_history_required_N_models", "d": "mc_history_required_d_models"}[design]
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=260)
    plt.close(fig)

