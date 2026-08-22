"""Compatibility and statistical helpers for the final Monte Carlo extension.

The base DGP and estimator live in ``finance_mc_core.py``.  This module contains
only the reusable post-processing routines needed by the asset-dimension and
persistence comparative statics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_snr(sr_annual: float, periods_per_year: int = 12) -> float:
    """Population maximum squared Sharpe per period."""
    return sr_annual**2 / periods_per_year


def seed_for(seed: int, block: int, T: int, rep: int, signal_id: int = 0) -> int:
    """Deterministic non-overlapping random-number stream identifier."""
    return int(seed + block * 1_000_000_000 + signal_id * 100_000_000 + T * 10_000 + rep)


def quantile_summary(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, object]] = {
        "sharpe_median": ("sharpe_annual", "median"),
        "sharpe_p10": ("sharpe_annual", lambda x: np.quantile(x, 0.10)),
        "sharpe_p90": ("sharpe_annual", lambda x: np.quantile(x, 0.90)),
        "recovery_median": ("sharpe2_recovery", "median"),
        "recovery_p10": ("sharpe2_recovery", lambda x: np.quantile(x, 0.10)),
        "recovery_p90": ("sharpe2_recovery", lambda x: np.quantile(x, 0.90)),
        "relative_shortfall_median": ("relative_shortfall", "median"),
        "relative_shortfall_mean": ("relative_shortfall", "mean"),
        "n_features": ("n_features", "median"),
        "replications": ("rep", "nunique"),
    }
    if "oracle_sr_annual" in df.columns and "oracle_sr_annual" not in groups:
        aggregations["oracle_sr_annual"] = ("oracle_sr_annual", "first")
    if "portfolio_snr" in df.columns and "portfolio_snr" not in groups:
        aggregations["portfolio_snr"] = ("portfolio_snr", "first")
    if "ridge_scale" in df.columns and "ridge_scale" not in groups:
        aggregations["ridge_scale"] = ("ridge_scale", "first")
    return df.groupby(groups).agg(**aggregations).reset_index()


def bootstrap_exponent(df: pd.DataFrame, theory: float, n_tail: int, n_boot: int,
                       seed: int, value_col: str = "relative_shortfall") -> dict[str, float]:
    """Bootstrap the log-log tail exponent using cell medians."""
    ts = np.array(sorted(df["T"].unique()))[-n_tail:]
    values = {t: df.loc[df["T"].eq(t), value_col].to_numpy(dtype=float) for t in ts}
    def exponent(medians: np.ndarray) -> float:
        return float(-np.polyfit(np.log(ts), np.log(medians), 1)[0])
    point = exponent(np.array([np.median(values[t]) for t in ts]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        medians = [np.median(rng.choice(values[t], size=values[t].size, replace=True)) for t in ts]
        boot[b] = exponent(np.asarray(medians))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"tail_points": int(n_tail), "first_T": int(ts[0]), "last_T": int(ts[-1]),
            "theory_exponent": float(theory), "empirical_exponent": point,
            "ci_low": float(lo), "ci_high": float(hi)}


def rolling_exponents(df: pd.DataFrame, theory: float, window: int, n_boot: int,
                      seed: int, value_col: str = "relative_shortfall") -> pd.DataFrame:
    """Rolling log-log exponents with a cell bootstrap."""
    ts_all = np.array(sorted(df["T"].unique()))
    rows: list[dict] = []
    for start in range(0, len(ts_all) - window + 1):
        ts = ts_all[start : start + window]
        cell = df[df["T"].isin(ts)]
        values = {t: cell.loc[cell["T"].eq(t), value_col].to_numpy(dtype=float) for t in ts}
        def exponent(medians: np.ndarray) -> float:
            return float(-np.polyfit(np.log(ts), np.log(medians), 1)[0])
        point = exponent(np.array([np.median(values[t]) for t in ts]))
        rng = np.random.default_rng(seed + start)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            medians = [np.median(rng.choice(values[t], size=values[t].size, replace=True)) for t in ts]
            boot[b] = exponent(np.asarray(medians))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"start_T": int(ts[0]), "endpoint_T": int(ts[-1]),
                     "window_points": int(window), "theory_exponent": float(theory),
                     "empirical_exponent": point, "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows)
