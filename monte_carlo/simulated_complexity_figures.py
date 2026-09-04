#!/usr/bin/env python3
"""Paper-ready simulated figures for the portfolio-complexity paper.

The simulation is deliberately transparent. Returns live in an orthogonal
"economic eigenbasis" with population second moment

    A = diag(mu_j),       mu_j ~ j^{-b}.

The population response-one optimum satisfies the source condition

    w*_j = mu_j^{(r-1)/2} h_j,

with h_j ~ j^{-h_decay}. The mean is chosen as m = A w*, and the return
covariance is Sigma = A - m m', so E[R R'] = A exactly. Scaling w* fixes the
population maximum Sharpe ratio.

The complexity-indexed experiment uses a nested spectral sieve: a model of
complexity C can use the first C economic eigendirections. With A known, the
estimated policy uses the sample mean in those directions. This produces the
clean finite-sample competition central to the paper:

    approximation error decreases with C,
    estimation error grows approximately like C/T.

The figures are illustrative Monte Carlo evidence, not claims that the exact
DGP-specific Sharpe ratio must be globally concave for every economy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SEED = 20260904


def paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=["0.0", "0.30", "0.50", "0.68", "0.82"]),
    })


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    fig.savefig(outdir / f"{stem}.svg")
    plt.close(fig)


def spectral_dgp(n=400, b=1.10, r=1.10, h_decay=0.51, target_sr=2.0):
    j = np.arange(1, n + 1, dtype=float)
    mu = j ** (-b)
    h = j ** (-h_decay)
    w_raw = mu ** ((r - 1.0) / 2.0) * h
    s_target = target_sr**2 / (1.0 + target_sr**2)
    scale = np.sqrt(s_target / np.sum(mu * w_raw**2))
    w_star = scale * w_raw
    mean = mu * w_star
    sigma = np.diag(mu) - np.outer(mean, mean)
    min_eval = np.linalg.eigvalsh(sigma)[0]
    if min_eval <= 0:
        raise RuntimeError(f"Return covariance is not positive definite: {min_eval}")
    return mu, w_star, mean, sigma


def complexity_grid(c_max: int, n_grid: int = 90) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, c_max, n_grid)).astype(int))


def oracle_sharpe_by_complexity(mu, w_star, mean, c_grid):
    cum_mean = np.cumsum(w_star * mean)
    cum_second = np.cumsum(mu * w_star**2)
    out = []
    for c in c_grid:
        m = cum_mean[c - 1]
        variance = cum_second[c - 1] - m**2
        out.append(m / np.sqrt(max(variance, 1e-15)))
    return np.asarray(out)


def learned_sharpe_curves(t_values, reps, n=400, b=1.10, r=1.10,
                           h_decay=0.51, target_sr=2.0, c_max=200,
                           seed=SEED):
    mu, w_star, mean, sigma = spectral_dgp(n, b, r, h_decay, target_sr)
    c_grid = complexity_grid(min(c_max, n), 90)
    oracle = oracle_sharpe_by_complexity(mu, w_star, mean, c_grid)
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(seed)
    curves, standard_errors = {}, {}

    for t in t_values:
        noise = rng.standard_normal((reps, n)) @ chol.T / np.sqrt(float(t))
        mean_hat = mean + noise
        w_hat = mean_hat / mu
        cumulative_mean = np.cumsum(w_hat * mean, axis=1)
        cumulative_second = np.cumsum(w_hat**2 * mu, axis=1)
        sr = np.empty((reps, len(c_grid)))
        for k, c in enumerate(c_grid):
            m = cumulative_mean[:, c - 1]
            variance = cumulative_second[:, c - 1] - m**2
            sr[:, k] = m / np.sqrt(np.maximum(variance, 1e-15))
        curves[int(t)] = sr.mean(axis=0)
        standard_errors[int(t)] = sr.std(axis=0, ddof=1) / np.sqrt(reps)

    return c_grid, curves, standard_errors, oracle, (mu, w_star, mean, sigma)


def in_sample_vs_out_of_sample(t, reps, n=150, b=1.10, r=1.10,
                                h_decay=0.51, target_sr=2.0, c_max=100,
                                seed=SEED + 17):
    """Exact empirical response-one optimum in nested C-dimensional classes."""
    mu, _w_star, mean, sigma = spectral_dgp(n, b, r, h_decay, target_sr)
    c_grid = complexity_grid(min(c_max, n, t - 2), 38)
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(seed)
    sr_in = np.empty((reps, len(c_grid)))
    sr_out = np.empty_like(sr_in)

    for q in range(reps):
        returns = rng.standard_normal((t, n)) @ chol.T + mean
        for k, c in enumerate(c_grid):
            x = returns[:, :c]
            mean_hat = x.mean(axis=0)
            second_hat = x.T @ x / float(t)
            w_hat = np.linalg.solve(second_hat + 1e-10 * np.eye(c), mean_hat)
            rp = x @ w_hat
            sr_in[q, k] = rp.mean() / rp.std(ddof=1)
            m = mean[:c] @ w_hat
            variance = w_hat @ sigma[:c, :c] @ w_hat
            sr_out[q, k] = m / np.sqrt(max(variance, 1e-15))

    return c_grid, sr_in.mean(axis=0), sr_out.mean(axis=0)


def figure_regret_decomposition(outdir, b, r):
    br, t, a0, b0 = b * r, 120, 1.2, 1.0
    c = np.geomspace(1.0, 100.0, 500)
    approx = a0 * c ** (-br)
    estimation = b0 * c / t
    total = approx + estimation
    c_star = (br * a0 * t / b0) ** (1.0 / (br + 1.0))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(c, approx, lw=1.7, ls="--", label="Approximation")
    ax.plot(c, estimation, lw=1.7, ls=":", label="Estimation")
    ax.plot(c, total, lw=2.2, label="Total finite-sample regret")
    ax.axvline(c_star, lw=1.0, ls="-.", alpha=0.8)
    ax.annotate(r"$C_T^\star$", xy=(c_star, np.interp(c_star, c, total)),
                xytext=(c_star * 1.25, np.interp(c_star, c, total) * 1.8),
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Effective portfolio complexity $C$")
    ax.set_ylabel("Portfolio regret")
    ax.set_title("The finite-sample law of portfolio learnability")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "fig_sim_01_regret_decomposition")


def figure_oracle_vs_learned(outdir, reps, b, r, h_decay):
    t_values = (30, 120, 480)
    c_grid, curves, _se, oracle, _ = learned_sharpe_curves(
        t_values, reps, b=b, r=r, h_decay=h_decay, seed=SEED)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(c_grid, oracle, lw=2.3, label="Oracle")
    for t in t_values:
        curve = curves[t]
        ax.plot(c_grid, curve, lw=1.7, label=fr"Learned, $T={t}$")
        i = int(np.nanargmax(curve))
        ax.plot(c_grid[i], curve[i], marker="o", ms=4.5, linestyle="None")
    ax.set_xscale("log")
    ax.set_xlabel(r"Portfolio complexity $C$")
    ax.set_ylabel("Population Sharpe ratio")
    ax.set_title("Oracle complexity versus finite-sample learnability")
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, outdir, "fig_sim_02_oracle_vs_learned_sharpe")


def figure_scaling_law(outdir, reps, b, r, h_decay):
    t_values = np.asarray([30, 45, 60, 90, 120, 180, 240, 360, 480, 720])
    c_grid, curves, _se, _oracle, _ = learned_sharpe_curves(
        t_values, reps, b=b, r=r, h_decay=h_decay, seed=SEED + 3)
    c_star = np.asarray([c_grid[np.nanargmax(curves[int(t)])] for t in t_values])
    empirical_slope, empirical_intercept = np.polyfit(np.log(t_values), np.log(c_star), 1)
    alpha = b * r + 2.0 * h_decay - 1.0
    theory_slope = 1.0 / (alpha + 1.0)
    theory_level = np.exp(np.mean(np.log(c_star) - theory_slope * np.log(t_values)))
    empirical_fit = np.exp(empirical_intercept) * t_values**empirical_slope
    theory_fit = theory_level * t_values**theory_slope
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.scatter(t_values, c_star, s=28, label="Monte Carlo optimum", zorder=3)
    ax.plot(t_values, empirical_fit, lw=1.8,
            label=fr"Monte Carlo fit: slope $={empirical_slope:.2f}$")
    ax.plot(t_values, theory_fit, lw=1.6, ls="--",
            label=fr"Spectral prediction: slope $={theory_slope:.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Training history $T$")
    ax.set_ylabel(r"Sharpe-optimal complexity $\widehat C_T^\star$")
    ax.set_title("More data move the complexity frontier outward")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "fig_sim_03_complexity_scaling_law")
    return t_values, c_star, empirical_slope, theory_slope


def figure_in_vs_out(outdir, reps, b, r, h_decay):
    c_grid, sr_in, sr_out = in_sample_vs_out_of_sample(
        120, max(250, reps // 5), b=b, r=r, h_decay=h_decay)
    i = int(np.nanargmax(sr_out))
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.plot(c_grid, sr_in, lw=1.8, ls="--", label="In-sample Sharpe")
    ax.plot(c_grid, sr_out, lw=2.1, label="Out-of-sample Sharpe")
    ax.axvline(c_grid[i], lw=1.0, ls=":", alpha=0.8)
    ax.annotate(r"$\widehat C_T^\star$", xy=(c_grid[i], sr_out[i]),
                xytext=(c_grid[i] * 1.7, sr_out[i] * 0.95),
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xscale("log")
    ax.set_xlabel(r"Portfolio complexity $C$")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("More fit is not more learning")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "fig_sim_04_in_sample_vs_out_of_sample_sharpe")


def figure_nominal_vs_effective(outdir, b):
    p_values = np.unique(np.round(np.geomspace(2, 10000, 160)).astype(int))
    lambdas = (0.10, 0.03, 0.01)
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for lam in lambdas:
        c_eff = []
        for p in p_values:
            j = np.arange(1, p + 1, dtype=float)
            mu = j ** (-b)
            c_eff.append(np.sum(mu / (mu + lam)))
        ax.plot(p_values, c_eff, lw=1.8, label=fr"$\lambda={lam:g}$")
    ax.plot(p_values, p_values, lw=1.1, ls=":", label=r"$C=P$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Nominal complexity $P$")
    ax.set_ylabel(r"Effective complexity $\mathcal{N}(\lambda)$")
    ax.set_title("Nominal dimension and statistically active dimension differ")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "fig_sim_05_nominal_vs_effective_complexity")


def write_summary(outdir, t_values, c_star, empirical_slope, theory_slope):
    lines = ["T,C_star", *[f"{int(t)},{int(c)}" for t, c in zip(t_values, c_star)]]
    (outdir / "fig_sim_03_scaling_points.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "README.txt").write_text(
        "SIMULATED PORTFOLIO COMPLEXITY FIGURES\n\n"
        "These are illustrative Monte Carlo figures for the paper.\n"
        "The oracle curve is monotone because the spectral policy classes are nested.\n"
        "The learned curves are finite-sample Monte Carlo performance, not a theorem\n"
        "that DGP-specific Sharpe must be globally concave.\n\n"
        f"Monte Carlo scaling slope: {empirical_slope:.4f}\n"
        f"Spectral scaling prediction: {theory_slope:.4f}\n\n"
        "Each figure is saved as PDF (Overleaf), PNG, and SVG.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).resolve().parent / "figures_simulated")
    parser.add_argument("--reps", type=int, default=2500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    reps = 350 if args.quick else args.reps
    b, r, h_decay = 1.10, 1.10, 0.51
    paper_style()
    figure_regret_decomposition(args.outdir, b, r)
    figure_oracle_vs_learned(args.outdir, reps, b, r, h_decay)
    t_values, c_star, empirical_slope, theory_slope = figure_scaling_law(
        args.outdir, reps, b, r, h_decay)
    figure_in_vs_out(args.outdir, reps, b, r, h_decay)
    figure_nominal_vs_effective(args.outdir, b)
    write_summary(args.outdir, t_values, c_star, empirical_slope, theory_slope)
    print(f"Wrote simulated figures to {args.outdir}")
    print(f"Monte Carlo scaling slope: {empirical_slope:.3f}")
    print(f"Spectral scaling prediction: {theory_slope:.3f}")


if __name__ == "__main__":
    main()
