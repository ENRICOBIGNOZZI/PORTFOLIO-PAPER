from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

KernelName = Literal["matern", "rbf", "linear"]


@dataclass(frozen=True)
class KernelSpec:
    name: KernelName = "matern"
    nu: float = 1.5
    ell: float | None = None
    max_features: int = 2048
    seed: int = 17


@dataclass
class ManagedFeatureCache:
    dates: pd.DatetimeIndex
    test_mask: np.ndarray
    returns_count: np.ndarray
    cos_part: np.ndarray | None = None
    sin_part: np.ndarray | None = None
    linear_part: np.ndarray | None = None
    kernel: KernelSpec | None = None
    feature_names: tuple[str, ...] = ()
    ell: float | None = None

    def matrix(self, p: int | None = None) -> np.ndarray:
        if self.linear_part is not None:
            return self.linear_part.astype(np.float64, copy=False)
        if self.cos_part is None or self.sin_part is None:
            raise ValueError("RFF cache is incomplete")
        m_max = self.cos_part.shape[1]
        if p is None:
            m = m_max
        else:
            if p < 2:
                raise ValueError("p must be at least 2 for paired cosine/sine RFF")
            m = min(m_max, int(p) // 2)
        # Paired features approximate k(x,z) = E cos(w'(x-z)).
        # Using 1/sqrt(m) keeps k(x,x) approximately one as P changes.
        return np.concatenate(
            [self.cos_part[:, :m], self.sin_part[:, :m]], axis=1
        ).astype(np.float64, copy=False) / np.sqrt(float(m))


@dataclass
class RollingCurve:
    lambdas: np.ndarray
    avg_complexity: np.ndarray
    oos_sharpe: np.ndarray
    mean_is_sharpe: np.ndarray
    oos_returns: np.ndarray
    is_sharpe_by_date: np.ndarray
    complexity_by_date: np.ndarray
    test_dates: pd.DatetimeIndex
    window: int

    @property
    def best_index(self) -> int:
        x = np.asarray(self.oos_sharpe)
        if not np.isfinite(x).any():
            raise ValueError("No finite OOS Sharpe ratios")
        return int(np.nanargmax(x))

    @property
    def best_complexity(self) -> float:
        return float(self.avg_complexity[self.best_index])

    @property
    def best_sharpe(self) -> float:
        return float(self.oos_sharpe[self.best_index])

    @property
    def interior_maximum(self) -> bool:
        i = self.best_index
        return 0 < i < len(self.lambdas) - 1


def extract_feature_names(features: pd.DataFrame | pd.Series | Iterable[str]) -> list[str]:
    if isinstance(features, pd.DataFrame):
        if "features" in features.columns:
            vals = features["features"]
        elif len(features.columns) == 1:
            vals = features.iloc[:, 0]
        else:
            raise ValueError("features DataFrame must contain a 'features' column")
        return [str(x) for x in vals.dropna().tolist()]
    if isinstance(features, pd.Series):
        return [str(x) for x in features.dropna().tolist()]
    return [str(x) for x in features]


def _rank_center_month(block: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Cross-sectional percentile ranks in [-0.5, 0.5], missing -> 0.

    This uses only the contemporaneous cross section, so it is compatible with
    CTF temporal-integrity rules and avoids full-sample normalization leakage.
    """
    raw = block.loc[:, feature_names]
    if "excntry" in block.columns:
        ranked = raw.groupby(block["excntry"], sort=False).rank(axis=0, method="average", pct=True)
    else:
        ranked = raw.rank(axis=0, method="average", pct=True)
    zero = raw.eq(0).to_numpy(copy=False)
    x = ranked.to_numpy(dtype=np.float32, copy=True)
    x -= np.float32(0.5)
    # Match the CTF factor-ML convention: exact raw zeros remain a special value.
    x[zero] = np.float32(-0.5)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def estimate_median_lengthscale(
    chars: pd.DataFrame,
    feature_names: list[str],
    *,
    pretest_only: bool = True,
    max_months: int = 24,
    max_rows_per_month: int = 300,
    n_pairs: int = 10000,
    seed: int = 17,
) -> float:
    """Median-distance length scale using only pre-test observations by default."""
    df = chars
    if pretest_only and "ctff_test" in df.columns:
        df = df.loc[~df["ctff_test"].astype(bool)]
    if df.empty:
        raise ValueError("No observations available to estimate kernel length scale")

    dates = pd.to_datetime(df["eom"]).dropna().sort_values().unique()
    if len(dates) > max_months:
        idx = np.linspace(0, len(dates) - 1, max_months, dtype=int)
        dates = dates[idx]

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    for d in dates:
        b = df.loc[pd.to_datetime(df["eom"]) == pd.Timestamp(d)]
        if b.empty:
            continue
        x = _rank_center_month(b, feature_names)
        if len(x) > max_rows_per_month:
            take = rng.choice(len(x), size=max_rows_per_month, replace=False)
            x = x[take]
        pieces.append(x)
    if not pieces:
        raise ValueError("Could not form a feature sample for length-scale estimation")
    sample = np.concatenate(pieces, axis=0).astype(np.float64, copy=False)
    if len(sample) < 2:
        return 1.0

    m = min(n_pairs, max(1, len(sample) * 2))
    i = rng.integers(0, len(sample), size=m)
    j = rng.integers(0, len(sample), size=m)
    keep = i != j
    if not keep.any():
        return 1.0
    dist = np.linalg.norm(sample[i[keep]] - sample[j[keep]], axis=1)
    dist = dist[np.isfinite(dist) & (dist > 0)]
    if len(dist) == 0:
        return 1.0
    ell = float(np.median(dist))
    return max(ell, 1e-6)


def sample_spectral_frequencies(
    d: int,
    m: int,
    *,
    kernel: KernelName,
    ell: float,
    nu: float = 1.5,
    seed: int = 17,
) -> np.ndarray:
    """Sample nested spectral frequencies for RBF or Matérn kernels.

    For the standard Matérn covariance with smoothness nu and length scale ell,
    the spectral law is a multivariate Student-t with 2*nu degrees of freedom
    and scale ell^{-2} I.
    """
    rng = np.random.default_rng(seed)
    if kernel == "rbf":
        return rng.normal(size=(d, m)) / float(ell)
    if kernel == "matern":
        if nu <= 0:
            raise ValueError("Matérn nu must be positive")
        q = 2.0 * float(nu)
        z = rng.normal(size=(d, m))
        u = rng.chisquare(df=q, size=m)
        return z / np.sqrt(u / q)[None, :] / float(ell)
    raise ValueError(f"No spectral frequencies for kernel={kernel}")


def build_managed_feature_cache(
    chars: pd.DataFrame,
    feature_names: list[str],
    spec: KernelSpec,
    *,
    return_col: str = "ret_exc_lead1m",
    date_col: str = "eom",
    min_assets: int = 20,
    frequency_batch: int = 256,
    ell: float | None = None,
) -> ManagedFeatureCache:
    """Build monthly managed-payoff features directly from stock-level CTF data.

    At month t the scalar weight function is approximated by phi(x_it)' beta / N_t.
    Therefore the ex-post managed feature is
        F_t = N_t^{-1} sum_i r_{i,t+1} phi(x_it).
    The response-one estimator is ridge regression of 1 on F_t.
    """
    required = {date_col, return_col, *feature_names}
    missing = required.difference(chars.columns)
    if missing:
        raise ValueError(f"chars is missing required columns: {sorted(missing)}")

    dates_series = pd.to_datetime(chars[date_col])
    tmp = chars.copy(deep=False)
    # shallow copy, but replace only date vector through a helper column
    tmp = tmp.assign(_eom_kernel=dates_series)
    dates = pd.DatetimeIndex(sorted(tmp["_eom_kernel"].dropna().unique()))
    if len(dates) == 0:
        raise ValueError("No valid month-end dates")

    if "ctff_test" in tmp.columns:
        test_by_date = tmp.groupby("_eom_kernel", sort=False)["ctff_test"].max().astype(bool)
        test_mask = np.array([bool(test_by_date.get(d, False)) for d in dates], dtype=bool)
    else:
        test_mask = np.zeros(len(dates), dtype=bool)

    ell_used = ell if ell is not None else spec.ell
    if spec.name in {"matern", "rbf"} and ell_used is None:
        ell_used = estimate_median_lengthscale(tmp, feature_names, pretest_only=True, seed=spec.seed)

    counts = np.zeros(len(dates), dtype=np.int32)
    if spec.name == "linear":
        linear = np.zeros((len(dates), len(feature_names)), dtype=np.float32)
        cos_part = sin_part = None
        W = None
    else:
        m_max = max(1, int(spec.max_features) // 2)
        W = sample_spectral_frequencies(
            len(feature_names), m_max, kernel=spec.name, ell=float(ell_used), nu=spec.nu, seed=spec.seed
        ).astype(np.float32)
        cos_part = np.zeros((len(dates), m_max), dtype=np.float32)
        sin_part = np.zeros((len(dates), m_max), dtype=np.float32)
        linear = None

    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    for d, block in tmp.groupby("_eom_kernel", sort=True):
        d = pd.Timestamp(d)
        t = date_to_pos[d]
        x = _rank_center_month(block, feature_names)
        r = pd.to_numeric(block[return_col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(r)
        if int(valid.sum()) < min_assets:
            continue
        x = x[valid]
        r = r[valid]
        n = len(r)
        counts[t] = n

        if spec.name == "linear":
            linear[t] = (r @ x / n).astype(np.float32)
            continue

        assert W is not None and cos_part is not None and sin_part is not None
        for j0 in range(0, W.shape[1], frequency_batch):
            j1 = min(W.shape[1], j0 + frequency_batch)
            proj = x @ W[:, j0:j1]
            cos_part[t, j0:j1] = (r @ np.cos(proj) / n).astype(np.float32)
            sin_part[t, j0:j1] = (r @ np.sin(proj) / n).astype(np.float32)

    valid_month = counts >= min_assets
    if not valid_month.all():
        dates = dates[valid_month]
        test_mask = test_mask[valid_month]
        counts = counts[valid_month]
        if linear is not None:
            linear = linear[valid_month]
        if cos_part is not None:
            cos_part = cos_part[valid_month]
            sin_part = sin_part[valid_month]

    return ManagedFeatureCache(
        dates=dates,
        test_mask=test_mask,
        returns_count=counts,
        cos_part=cos_part,
        sin_part=sin_part,
        linear_part=linear,
        kernel=spec,
        feature_names=tuple(feature_names),
        ell=float(ell_used) if ell_used is not None else None,
    )


def lambda_grid_from_reference(
    F: np.ndarray,
    test_mask: np.ndarray,
    window: int,
    *,
    log10_min: float = -6.0,
    log10_max: float = 4.0,
    n: int = 61,
) -> np.ndarray:
    test_idx = np.flatnonzero(test_mask)
    test_idx = test_idx[test_idx >= window]
    if len(test_idx) == 0:
        raise ValueError(f"No test date has at least {window} prior months")
    t = int(test_idx[0])
    X = F[t - window : t]
    G = X @ X.T / float(window)
    scale = float(np.trace(G) / max(1, len(G)))
    scale = max(scale, 1e-12)
    return scale * np.logspace(log10_min, log10_max, int(n))


def _annualized_sharpe_matrix(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=1)
    out = np.full_like(mu, np.nan, dtype=np.float64)
    good = np.isfinite(sd) & (sd > 0)
    out[good] = np.sqrt(12.0) * mu[good] / sd[good]
    return out


def rolling_response_one_curve(
    F: np.ndarray,
    dates: pd.DatetimeIndex,
    test_mask: np.ndarray,
    *,
    window: int,
    lambdas: np.ndarray | None = None,
    common_start_index: int | None = None,
) -> RollingCurve:
    """One-step rolling response-one ridge evaluation using the dual Gram matrix.

    For each test month t, training uses only t-window,...,t-1. The current
    return vector enters only after beta is fixed, through the ex-post payoff F_t beta.
    """
    F = np.asarray(F, dtype=np.float64)
    lambdas = (
        lambda_grid_from_reference(F, test_mask, window)
        if lambdas is None
        else np.asarray(lambdas, dtype=np.float64)
    )
    idx = np.flatnonzero(test_mask)
    idx = idx[idx >= window]
    if common_start_index is not None:
        idx = idx[idx >= int(common_start_index)]
    if len(idx) < 3:
        raise ValueError("Too few eligible OOS test months")

    L = len(lambdas)
    oos = np.full((len(idx), L), np.nan, dtype=np.float64)
    is_sr = np.full((len(idx), L), np.nan, dtype=np.float64)
    comp = np.full((len(idx), L), np.nan, dtype=np.float64)

    ones = np.ones(window, dtype=np.float64)
    for k, t in enumerate(idx):
        X = F[t - window : t]
        G = X @ X.T / float(window)
        eig, U = np.linalg.eigh(G)
        eig = np.clip(eig, 0.0, None)
        u1 = U.T @ ones

        den = eig[:, None] + lambdas[None, :]
        inv_rhs = U @ (u1[:, None] / den)
        # beta = F_train' (G+lambda I)^-1 1 / T
        cross = (F[t] @ X.T) / float(window)
        oos[k] = cross @ inv_rhs

        fitted = U @ ((eig[:, None] / den) * u1[:, None])
        is_sr[k] = _annualized_sharpe_matrix(fitted)
        comp[k] = np.sum(eig[:, None] / den, axis=0)

    return RollingCurve(
        lambdas=lambdas,
        avg_complexity=np.nanmean(comp, axis=0),
        oos_sharpe=_annualized_sharpe_matrix(oos),
        mean_is_sharpe=np.nanmean(is_sr, axis=0),
        oos_returns=oos,
        is_sharpe_by_date=is_sr,
        complexity_by_date=comp,
        test_dates=dates[idx],
        window=int(window),
    )


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    *,
    block: int = 12,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving-block bootstrap CI for columns of monthly return series."""
    x = np.asarray(returns, dtype=np.float64)
    T, L = x.shape
    if T < max(6, block):
        nan = np.full(L, np.nan)
        return nan, nan
    rng = np.random.default_rng(seed)
    starts = np.arange(0, T - block + 1)
    n_blocks = int(np.ceil(T / block))
    boot = np.full((n_boot, L), np.nan, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        ii = np.concatenate([np.arange(s, s + block) for s in chosen])[:T]
        boot[b] = _annualized_sharpe_matrix(x[ii])
    lo = np.nanquantile(boot, alpha / 2.0, axis=0)
    hi = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)
    return lo, hi


def profiled_response_one_loss(returns: np.ndarray) -> np.ndarray:
    """Scale-profiled response-one loss for each return-series column.

    For a direction R, min_a E(1-aR)^2 = 1 - E[R]^2/E[R^2].
    This is scale invariant and is monotone in squared Sharpe ratio.
    """
    x = np.asarray(returns, dtype=np.float64)
    m = np.nanmean(x, axis=0)
    m2 = np.nanmean(x * x, axis=0)
    out = np.full_like(m, np.nan)
    good = np.isfinite(m2) & (m2 > 0)
    out[good] = 1.0 - (m[good] ** 2) / m2[good]
    return out
