"""CTF-compatible Matérn portfolio-kernel submission.

This file is intentionally self-contained and exposes the exact CTF main() signature.
It uses only information available at or before each decision date. The method is the
finite-RFF approximation of the same Matérn response-one portfolio kernel used in the
paper figures.

For paper inference and the full complexity sweeps, use run_paper_figures.py instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_names(features: pd.DataFrame) -> list[str]:
    if "features" in features.columns:
        return [str(x) for x in features["features"].dropna().tolist()]
    if len(features.columns) == 1:
        return [str(x) for x in features.iloc[:, 0].dropna().tolist()]
    raise ValueError("features must contain a 'features' column")


def _rank_center(block: pd.DataFrame, names: list[str]) -> np.ndarray:
    raw = block[names]
    if "excntry" in block.columns:
        ranked = raw.groupby(block["excntry"], sort=False).rank(method="average", pct=True)
    else:
        ranked = raw.rank(axis=0, method="average", pct=True)
    zero = raw.eq(0).to_numpy(copy=False)
    x = ranked.to_numpy(dtype=np.float32, copy=False)
    x -= np.float32(0.5)
    x[zero] = np.float32(-0.5)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _matern_frequencies(d: int, m: int, ell: float, nu: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = 2.0 * nu
    z = rng.normal(size=(d, m))
    u = rng.chisquare(q, size=m)
    return (z / np.sqrt(u / q)[None, :] / ell).astype(np.float32)


def _estimate_ell(chars: pd.DataFrame, names: list[str], first_test: pd.Timestamp, seed: int) -> float:
    eom = pd.to_datetime(chars["eom"])
    pre = chars.loc[eom < first_test]
    pre_eom = pd.to_datetime(pre["eom"])
    dates = pd.DatetimeIndex(sorted(pre_eom.dropna().unique()))
    if len(dates) > 18:
        dates = dates[np.linspace(0, len(dates) - 1, 18, dtype=int)]
    rng = np.random.default_rng(seed)
    xs = []
    for d in dates:
        b = pre.loc[pre_eom == d]
        if b.empty:
            continue
        x = _rank_center(b, names)
        if len(x) > 250:
            x = x[rng.choice(len(x), 250, replace=False)]
        xs.append(x)
    if not xs:
        return 1.0
    x = np.concatenate(xs, axis=0).astype(np.float64)
    n = min(8000, max(500, 2 * len(x)))
    i = rng.integers(0, len(x), n)
    j = rng.integers(0, len(x), n)
    keep = i != j
    dist = np.linalg.norm(x[i[keep]] - x[j[keep]], axis=1)
    dist = dist[(dist > 0) & np.isfinite(dist)]
    return float(np.median(dist)) if len(dist) else 1.0


def _phi(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    z = x @ W
    m = W.shape[1]
    return np.concatenate([np.cos(z), np.sin(z)], axis=1) / np.sqrt(float(m))


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    # Conservative settings selected a priori. No test-period tuning occurs here.
    seed = 17
    nu = 1.5
    P = 512
    m = P // 2
    train_T = 120
    lambda_multiplier = 1e-2
    min_assets = 20

    names = _feature_names(features)
    chars = chars.copy()
    chars["eom"] = pd.to_datetime(chars["eom"])
    all_dates = pd.DatetimeIndex(sorted(chars["eom"].dropna().unique()))
    if len(all_dates) == 0:
        return pd.DataFrame(columns=["id", "eom", "w"])

    has_test = "ctff_test" in chars.columns and chars["ctff_test"].fillna(False).astype(bool).any()
    if has_test:
        output_dates = pd.DatetimeIndex(
            sorted(chars.loc[chars["ctff_test"].fillna(False).astype(bool), "eom"].dropna().unique())
        )
    else:
        # The CTF validation sample may omit active test flags. Returning the
        # last date gives the validator a non-empty, causally formed portfolio
        # while keeping all preceding months available for training.
        output_dates = pd.DatetimeIndex([all_dates[-1]])

    first_test = output_dates[0]
    ell = _estimate_ell(chars, names, first_test, seed)
    W = _matern_frequencies(len(names), m, ell, nu, seed)
    output_set = set(pd.Timestamp(d) for d in output_dates)

    history: list[np.ndarray] = []
    out: list[pd.DataFrame] = []

    # Process dates sequentially. At a decision date t, beta is fit before F_t
    # is formed, so r_{t+1} is never used to choose the time-t portfolio weights.
    for d, b in chars.groupby("eom", sort=True):
        d = pd.Timestamp(d)
        b = b.reset_index(drop=True)
        x = _rank_center(b, names)
        ph = _phi(x, W).astype(np.float32)
        is_output = d in output_set

        if is_output:
            hist = np.asarray(history[-train_T:], dtype=np.float64)
            hist = hist[np.isfinite(hist).all(axis=1)] if len(hist) else hist
            if len(hist) < 12:
                w = np.zeros(len(b), dtype=np.float64)
            else:
                T = len(hist)
                G = hist @ hist.T / float(T)
                scale = max(float(np.trace(G) / max(1, T)), 1e-12)
                lam = lambda_multiplier * scale
                alpha = np.linalg.solve(G + lam * np.eye(T), np.ones(T))
                beta = hist.T @ alpha / float(T)
                w = (ph @ beta) / max(1, len(b))
            tmp = b[["id", "eom"]].copy()
            tmp["w"] = np.asarray(w, dtype=np.float64)
            out.append(tmp)

        # Only after output weights are fixed do we reveal the one-month-ahead return.
        r = pd.to_numeric(b["ret_exc_lead1m"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(r)
        if int(ok.sum()) >= min_assets:
            F_t = (r[ok] @ ph[ok] / ok.sum()).astype(np.float64)
        else:
            F_t = np.full(P, np.nan, dtype=np.float64)
        history.append(F_t)

    if not out:
        return pd.DataFrame(columns=["id", "eom", "w"])
    return pd.concat(out, ignore_index=True)[["id", "eom", "w"]].dropna()
