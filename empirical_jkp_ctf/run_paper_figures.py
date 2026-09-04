from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import (
    KernelSpec,
    ManagedFeatureCache,
    block_bootstrap_sharpe_ci,
    build_managed_feature_cache,
    extract_feature_names,
    profiled_response_one_loss,
    rolling_response_one_curve,
)


def savefig(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def curve_frame(curve, kernel: str, p: int) -> pd.DataFrame:
    lo, hi = block_bootstrap_sharpe_ci(curve.oos_returns, block=12, n_boot=300)
    return pd.DataFrame(
        {
            "kernel": kernel,
            "P": p,
            "window_T": curve.window,
            "lambda": curve.lambdas,
            "effective_complexity": curve.avg_complexity,
            "sharpe_oos": curve.oos_sharpe,
            "sharpe_is_mean": curve.mean_is_sharpe,
            "sharpe_oos_ci_lo": lo,
            "sharpe_oos_ci_hi": hi,
            "profiled_oos_loss": profiled_response_one_loss(curve.oos_returns),
        }
    )


def plot_fig1(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("effective_complexity")
    i = int(d["sharpe_oos"].to_numpy().argmax())
    x = d["effective_complexity"].to_numpy()
    y = d["sharpe_oos"].to_numpy()
    lo = d["sharpe_oos_ci_lo"].to_numpy()
    hi = d["sharpe_oos_ci_hi"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(x, y, marker="o", markersize=3, linewidth=1.4)
    ax.fill_between(x, lo, hi, alpha=0.15)
    ax.axvline(x[i], linestyle="--", linewidth=1.0)
    ax.scatter([x[i]], [y[i]], s=35, zorder=4)
    ax.set_xlabel("Effective portfolio complexity $C$")
    ax.set_ylabel("Out-of-sample Sharpe ratio")
    ax.set_title("Out-of-sample Sharpe across effective complexity")
    ax.grid(alpha=0.2)
    savefig(fig, out, "fig1_oos_sharpe_vs_effective_complexity")


def plot_fig2(summary: pd.DataFrame, out: Path) -> None:
    d = summary.sort_values("window_T")
    x = d["window_T"].to_numpy(dtype=float)
    y = d["C_star"].to_numpy(dtype=float)
    good = (x > 0) & (y > 0) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.loglog(x, y, marker="o", linewidth=1.4, label=r"$\widehat C_T^\star$")
    if good.sum() >= 2:
        slope, intercept = np.polyfit(np.log(x[good]), np.log(y[good]), 1)
        fit = np.exp(intercept) * x ** slope
        ax.loglog(x, fit, linestyle="--", linewidth=1.0, label=f"log-log slope = {slope:.2f}")
    ax.set_xlabel("Training history $T$ (months)")
    ax.set_ylabel(r"Sharpe-maximizing effective complexity $\widehat C_T^\star$")
    ax.set_title("Optimal effective complexity and sample size")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")
    savefig(fig, out, "fig2_optimal_complexity_vs_sample_size")


def plot_fig3(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("effective_complexity")
    x = d["effective_complexity"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(x, d["sharpe_is_mean"], marker="o", markersize=3, linewidth=1.3, label="In sample")
    ax.plot(x, d["sharpe_oos"], marker="s", markersize=3, linewidth=1.3, label="Out of sample")
    ax.set_xlabel("Effective portfolio complexity $C$")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("The complexity wedge")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    savefig(fig, out, "fig3_in_sample_vs_oos_sharpe")


def plot_fig4(summary: pd.DataFrame, out: Path) -> None:
    d = summary.sort_values("P")
    x = d["P"].to_numpy(dtype=float)
    c = d["C_star"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.loglog(x, x, linestyle="--", linewidth=1.0, label="Nominal complexity $P$")
    ax.loglog(x, c, marker="o", linewidth=1.4, label=r"Optimal effective complexity $\widehat C_T^\star$")
    ax.set_xlabel("Nominal feature dimension $P$")
    ax.set_ylabel("Complexity")
    ax.set_title("Large representations, finite active complexity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")
    savefig(fig, out, "fig4_nominal_vs_effective_complexity")


def plot_fig5(robust: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for name, d in robust.groupby("kernel", sort=False):
        d = d.sort_values("effective_complexity")
        ax.plot(d["effective_complexity"], d["sharpe_oos"], linewidth=1.2, label=name)
    ax.set_xlabel("Effective portfolio complexity $C$")
    ax.set_ylabel("Out-of-sample Sharpe ratio")
    ax.set_title("Kernel robustness")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    savefig(fig, out, "fig5_kernel_robustness")


def best_row(df: pd.DataFrame) -> pd.Series:
    d = df.loc[np.isfinite(df["sharpe_oos"])]
    if d.empty:
        raise ValueError("No finite OOS Sharpe ratio")
    return d.loc[d["sharpe_oos"].idxmax()]


def main() -> None:
    ap = argparse.ArgumentParser(description="JKP/CTF empirical complexity figures")
    ap.add_argument("--chars", type=Path, default=Path("data/raw/ctff_chars.parquet"))
    ap.add_argument("--features", type=Path, default=Path("data/raw/ctff_features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("outputs/jkp_ctf"))
    ap.add_argument("--primary-T", type=int, default=120)
    ap.add_argument("--T-grid", type=str, default="36,60,120,180,240")
    ap.add_argument("--P-grid", type=str, default="32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--P-max", type=int, default=4096)
    ap.add_argument("--nu", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    chars = pd.read_parquet(args.chars)
    features_df = pd.read_parquet(args.features)
    feature_names = extract_feature_names(features_df)

    T_grid = [int(x) for x in args.T_grid.split(",") if x.strip()]
    P_grid = [int(x) for x in args.P_grid.split(",") if x.strip()]
    max_T = max(T_grid + [args.primary_T])

    primary_spec = KernelSpec(name="matern", nu=args.nu, max_features=args.P_max, seed=args.seed)
    cache = build_managed_feature_cache(chars, feature_names, primary_spec)
    Fmax = cache.matrix(args.P_max)
    common_start = max_T

    # Figure 1 + 3: single Matérn, single T, complexity varied only through lambda.
    curve = rolling_response_one_curve(
        Fmax, cache.dates, cache.test_mask, window=args.primary_T, common_start_index=common_start
    )
    primary_df = curve_frame(curve, f"Matern-{args.nu:g}", args.P_max)
    primary_df.to_csv(args.out / "primary_complexity_curve.csv", index=False)
    plot_fig1(primary_df, args.out)
    plot_fig3(primary_df, args.out)

    # Figure 2: C*_T against T on a common OOS date set.
    T_rows = []
    T_curves = []
    for T in T_grid:
        c = rolling_response_one_curve(Fmax, cache.dates, cache.test_mask, window=T, common_start_index=common_start)
        d = curve_frame(c, f"Matern-{args.nu:g}", args.P_max)
        T_curves.append(d)
        b = best_row(d)
        T_rows.append(
            {
                "window_T": T,
                "C_star": float(b["effective_complexity"]),
                "SR_star_oos": float(b["sharpe_oos"]),
                "lambda_star": float(b["lambda"]),
                "interior_maximum": bool(0 < int(np.nanargmax(d["sharpe_oos"])) < len(d) - 1),
            }
        )
    T_summary = pd.DataFrame(T_rows)
    T_summary.to_csv(args.out / "optimal_complexity_by_T.csv", index=False)
    pd.concat(T_curves, ignore_index=True).to_csv(args.out / "complexity_curves_by_T.csv", index=False)
    plot_fig2(T_summary, args.out)

    # Figure 4: same Matérn family, nested RFF approximation P; lambda is reprofiled at each P.
    P_rows = []
    P_curves = []
    for P in P_grid:
        Fp = cache.matrix(P)
        c = rolling_response_one_curve(Fp, cache.dates, cache.test_mask, window=args.primary_T, common_start_index=common_start)
        d = curve_frame(c, f"Matern-{args.nu:g}", P)
        P_curves.append(d)
        b = best_row(d)
        P_rows.append(
            {
                "P": P,
                "C_star": float(b["effective_complexity"]),
                "SR_star_oos": float(b["sharpe_oos"]),
                "lambda_star": float(b["lambda"]),
                "C_over_P": float(b["effective_complexity"]) / float(P),
            }
        )
    P_summary = pd.DataFrame(P_rows)
    P_summary.to_csv(args.out / "nominal_vs_effective_complexity.csv", index=False)
    pd.concat(P_curves, ignore_index=True).to_csv(args.out / "complexity_curves_by_P.csv", index=False)
    plot_fig4(P_summary, args.out)

    # Figure 5: robustness is deliberately separate from the headline complexity experiment.
    robust_frames = [primary_df]
    if not args.skip_robustness:
        specs = [
            KernelSpec(name="matern", nu=0.5, max_features=args.P_max, seed=args.seed),
            KernelSpec(name="matern", nu=2.5, max_features=args.P_max, seed=args.seed),
            KernelSpec(name="rbf", max_features=args.P_max, seed=args.seed),
            KernelSpec(name="linear", max_features=len(feature_names), seed=args.seed),
        ]
        for spec in specs:
            # Use the primary pre-test median length scale for stationary kernels so the
            # robustness exercise does not retune geometry on the test sample.
            ccache = build_managed_feature_cache(chars, feature_names, spec, ell=cache.ell)
            F = ccache.matrix(args.P_max if spec.name != "linear" else None)
            c = rolling_response_one_curve(
                F, ccache.dates, ccache.test_mask, window=args.primary_T, common_start_index=common_start
            )
            label = "Linear" if spec.name == "linear" else ("RBF" if spec.name == "rbf" else f"Matern-{spec.nu:g}")
            robust_frames.append(curve_frame(c, label, F.shape[1]))
    robust = pd.concat(robust_frames, ignore_index=True)
    robust.to_csv(args.out / "kernel_robustness_curves.csv", index=False)
    plot_fig5(robust, args.out)

    best = best_row(primary_df)
    metadata = {
        "dataset": "Jensen-Kelly-Pedersen Common Task Framework (WRDS CTF tables)",
        "kernel": asdict(primary_spec),
        "estimated_lengthscale": cache.ell,
        "n_characteristics": len(feature_names),
        "n_months": len(cache.dates),
        "test_months": int(cache.test_mask.sum()),
        "primary_T": args.primary_T,
        "primary_P": args.P_max,
        "primary_C_star": float(best["effective_complexity"]),
        "primary_SR_star_oos": float(best["sharpe_oos"]),
        "primary_lambda_star": float(best["lambda"]),
        "primary_max_is_interior": bool(
            0 < int(np.nanargmax(primary_df["sharpe_oos"].to_numpy())) < len(primary_df) - 1
        ),
        "note": "C* is the ex-post OOS Sharpe maximizer of the displayed complexity curve; no interior maximum is imposed by code.",
    }
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
