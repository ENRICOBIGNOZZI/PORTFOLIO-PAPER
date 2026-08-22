#!/usr/bin/env python3
"""Finance-facing Monte Carlo for Learning Portfolio Decisions.

The script implements two experiments:
  1. a correctly specified linear conditional factor-premium economy;
  2. a three-dimensional nonlinear Sobolev economy with fixed smoothness/source
     parameters and a known population Sharpe ratio.

Each sample-size cell uses independent simulated market histories. Evaluation is
at the population level under the known DGP, so reported Sharpe ratios do not
contain an additional test-sample draw.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Avoid BLAS oversubscription when joblib launches worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from scipy.special import eval_legendre, ndtr, ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


@dataclass(frozen=True)
class Economy:
    B: np.ndarray
    idio_sd: np.ndarray
    rho: float = 0.70
    periods_per_year: int = 12

    @property
    def N(self) -> int:
        return int(self.B.shape[0])

    @property
    def K(self) -> int:
        return int(self.B.shape[1])

    @property
    def Omega(self) -> np.ndarray:
        return self.B @ self.B.T + np.diag(self.idio_sd ** 2)

    @property
    def A(self) -> np.ndarray:
        """Map factor premia into the conditional tangency policy."""
        return np.linalg.solve(self.Omega, self.B)

    @property
    def H(self) -> np.ndarray:
        """Factor-space managed-opportunity matrix B' Omega^{-1} B."""
        return self.B.T @ self.A


ECON = Economy(
    B=np.array(
        [
            [0.70, 0.15, 0.05],
            [0.58, -0.12, 0.22],
            [0.24, 0.66, 0.08],
            [-0.08, 0.54, 0.31],
            [0.16, 0.22, 0.68],
            [0.32, -0.04, 0.59],
        ],
        dtype=float,
    ),
    idio_sd=np.array([0.48, 0.52, 0.47, 0.55, 0.49, 0.57], dtype=float),
    rho=0.55,
    periods_per_year=12,
)

D = 3
TARGET_ANNUAL_SR = 1.50
S = 2.0
R_SOURCE = 1.5
B_SPECTRAL = 2.0 * S / D
THEORY_RATE = 2.0 * S * R_SOURCE / (2.0 * S * R_SOURCE + D)
THEORY_LAMBDA_EXP = 2.0 * S / (2.0 * S * R_SOURCE + D)


def target_m_from_annual_sharpe(sr_annual: float, periods_per_year: int) -> float:
    sr_period = sr_annual / math.sqrt(periods_per_year)
    return sr_period**2 / (1.0 + sr_period**2)


def annual_sharpe_from_m(m: float, periods_per_year: int) -> float:
    if not (0.0 < m < 1.0):
        raise ValueError(f"m must lie in (0,1), got {m}")
    return math.sqrt(periods_per_year * m / (1.0 - m))


TARGET_M = target_m_from_annual_sharpe(TARGET_ANNUAL_SR, ECON.periods_per_year)


# ---------------------------------------------------------------------------
# State and returns
# ---------------------------------------------------------------------------
def simulate_state(T: int, d: int, rho: float, rng: np.random.Generator,
                   burn: int = 300) -> np.ndarray:
    """Observed persistent state Z_t in (0,1)^d with uniform stationary marginals.

    The recursion is written directly in the observed state:
        Z_{t+1}=Phi(rho Phi^{-1}(Z_t)+sqrt(1-rho^2) eta_{t+1}).
    No second latent-state symbol is needed in the paper.
    """
    n = T + burn
    z = np.empty((n, d), dtype=float)
    z[0] = rng.uniform(1e-6, 1.0 - 1e-6, size=d)
    scale = math.sqrt(1.0 - rho**2)
    for t in range(1, n):
        latent = rho * ndtri(np.clip(z[t - 1], 1e-10, 1.0 - 1e-10))
        latent += scale * rng.standard_normal(d)
        z[t] = ndtr(latent)
    return z[burn:]


def sample_factors(lambda_z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw factors with E[F|Z]=lambda(Z) and E[FF'|Z]=I_K."""
    T, K = lambda_z.shape
    norms = np.linalg.norm(lambda_z, axis=1)
    max_norm = float(norms.max())
    if max_norm >= 0.96:
        raise ValueError(f"factor premia violate I-lambda lambda' >= 0: max norm={max_norm:.4f}")

    eps = rng.standard_normal((T, K))
    out = eps.copy()
    nz = norms > 1e-14
    if np.any(nz):
        u = lambda_z[nz] / norms[nz, None]
        projection = np.sum(u * eps[nz], axis=1)
        adjustment = np.sqrt(1.0 - norms[nz] ** 2) - 1.0
        out[nz] += (adjustment * projection)[:, None] * u
    return lambda_z + out


def sample_returns(lambda_z: np.ndarray, rng: np.random.Generator,
                   econ: Economy = ECON) -> np.ndarray:
    F = sample_factors(lambda_z, rng)
    u = rng.standard_normal((lambda_z.shape[0], econ.N)) * econ.idio_sd[None, :]
    return F @ econ.B.T + u


# ---------------------------------------------------------------------------
# Polynomial and Sobolev features
# ---------------------------------------------------------------------------
def total_degree_indices(d: int, degree: int) -> list[tuple[int, ...]]:
    return [a for a in itertools.product(range(degree + 1), repeat=d) if sum(a) <= degree]


def legendre_features(z: np.ndarray, degree: int) -> np.ndarray:
    """Orthonormal shifted-Legendre tensor basis up to a total degree."""
    z = np.asarray(z, dtype=float)
    x = 2.0 * z - 1.0
    one_d = np.empty((z.shape[0], z.shape[1], degree + 1), dtype=float)
    for q in range(degree + 1):
        one_d[:, :, q] = math.sqrt(2 * q + 1) * eval_legendre(q, x)
    idx = total_degree_indices(z.shape[1], degree)
    out = np.ones((z.shape[0], len(idx)), dtype=float)
    for j, alpha in enumerate(idx):
        for ell, q in enumerate(alpha):
            out[:, j] *= one_d[:, ell, q]
    return out


def canonical_frequency(k: tuple[int, ...]) -> bool:
    for value in k:
        if value != 0:
            return value > 0
    return False


def fourier_modes(d: int, n_pairs: int) -> list[np.ndarray]:
    """Return the lowest-frequency representatives of k and -k."""
    radius = 1
    candidates: list[tuple[int, tuple[int, ...]]] = []
    while len(candidates) < n_pairs:
        candidates.clear()
        for k in itertools.product(range(-radius, radius + 1), repeat=d):
            if all(v == 0 for v in k) or not canonical_frequency(k):
                continue
            norm2 = sum(v * v for v in k)
            if norm2 <= radius * radius:
                candidates.append((norm2, k))
        radius += 1
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [np.asarray(k, dtype=float) for _, k in candidates[:n_pairs]]


def fourier_l2_features(z: np.ndarray, modes: list[np.ndarray]) -> np.ndarray:
    """Real L2-orthonormal Fourier basis: constant, then cosine/sine pairs."""
    z = np.asarray(z, dtype=float)
    out = np.empty((z.shape[0], 1 + 2 * len(modes)), dtype=float)
    out[:, 0] = 1.0
    col = 1
    for k in modes:
        phase = 2.0 * np.pi * (z @ k)
        out[:, col] = math.sqrt(2.0) * np.cos(phase)
        out[:, col + 1] = math.sqrt(2.0) * np.sin(phase)
        col += 2
    return out


def fourier_eigenvalues(modes: list[np.ndarray], s: float) -> np.ndarray:
    tau = np.empty(1 + 2 * len(modes), dtype=float)
    tau[0] = 1.0
    col = 1
    for k in modes:
        value = (1.0 + float(k @ k)) ** (-s)
        tau[col] = value
        tau[col + 1] = value
        col += 2
    return tau


def sobolev_h_features(z: np.ndarray, modes: list[np.ndarray], s: float,
                       include_constant: bool = False) -> tuple[np.ndarray, np.ndarray]:
    phi = fourier_l2_features(z, modes)
    tau = fourier_eigenvalues(modes, s)
    psi = phi * np.sqrt(tau)[None, :]
    if include_constant:
        return psi, tau
    return psi[:, 1:], tau[1:]


def flexible_features(z: np.ndarray, modes: list[np.ndarray], s: float) -> np.ndarray:
    """Finite linear component plus nonlinear Sobolev component."""
    linear = legendre_features(z, degree=1)
    nonlinear, _ = sobolev_h_features(z, modes, s, include_constant=False)
    return np.column_stack([linear, nonlinear])


# ---------------------------------------------------------------------------
# Conditional factor premia and oracle policy
# ---------------------------------------------------------------------------
LINEAR_L = np.array(
    [
        [0.72, -0.30, 0.16],
        [-0.18, 0.64, 0.26],
        [0.22, 0.12, 0.58],
    ],
    dtype=float,
)


def linear_scale(econ: Economy = ECON) -> float:
    # q_j=sqrt(3)(2Z_j-1) are orthonormal, so E lambda'Hlambda=tr(L'HL).
    base_m = float(np.trace(LINEAR_L.T @ econ.H @ LINEAR_L))
    return math.sqrt(TARGET_M / base_m)


LINEAR_SCALE = linear_scale()


def lambda_linear(z: np.ndarray, econ: Economy = ECON) -> np.ndarray:
    q = legendre_features(z, degree=1)[:, 1:]
    return LINEAR_SCALE * (q @ LINEAR_L.T)


TARGET_PAIRS = 90
TARGET_MODES = fourier_modes(D, TARGET_PAIRS)
TARGET_TAU = fourier_eigenvalues(TARGET_MODES, S)


def nonlinear_source_coefficients(n_features: int, K: int, seed: int = 20260818,
                                  delta: float = 0.04) -> np.ndarray:
    """Fixed near-boundary source coefficients c_j in factor space."""
    rng = np.random.default_rng(seed)
    c = np.zeros((n_features, K), dtype=float)
    for j in range(1, n_features):
        direction = rng.standard_normal(K)
        direction /= np.linalg.norm(direction)
        magnitude = j ** (-0.5 - delta)
        c[j] = magnitude * direction
    # Make the first low-frequency modes economically interpretable and stable.
    anchors = np.array(
        [
            [1.00, 0.20, -0.10],
            [0.10, 0.90, 0.20],
            [-0.20, 0.15, 0.85],
            [0.55, -0.35, 0.25],
            [0.20, 0.50, -0.40],
            [-0.35, 0.20, 0.45],
        ]
    )
    c[1 : 1 + anchors.shape[0]] = anchors
    c /= np.linalg.norm(c)
    return c


SOURCE_C = nonlinear_source_coefficients(TARGET_TAU.size, ECON.K)
BASE_NONLINEAR_L2_COEF = (TARGET_TAU ** (R_SOURCE / 2.0))[:, None] * SOURCE_C


def nonlinear_scale(econ: Economy = ECON) -> float:
    base_m = 0.0
    for coeff in BASE_NONLINEAR_L2_COEF:
        base_m += float(coeff @ econ.H @ coeff)
    return math.sqrt(TARGET_M / base_m)


NONLINEAR_SCALE = nonlinear_scale()
NONLINEAR_L2_COEF = NONLINEAR_SCALE * BASE_NONLINEAR_L2_COEF


def lambda_nonlinear(z: np.ndarray) -> np.ndarray:
    phi = fourier_l2_features(z, TARGET_MODES)
    return phi @ NONLINEAR_L2_COEF


def true_policy(lambda_z: np.ndarray, econ: Economy = ECON) -> np.ndarray:
    return lambda_z @ econ.A.T


# ---------------------------------------------------------------------------
# Direct policy estimator and population metrics
# ---------------------------------------------------------------------------
def fit_direct_policy(features: np.ndarray, returns: np.ndarray, ridge: float) -> np.ndarray:
    """Solve min mean(1-w(Z)'R)^2 + ridge ||w||^2 in the supplied feature basis."""
    T, J = features.shape
    N = returns.shape[1]
    G = (returns[:, :, None] * features[:, None, :]).reshape(T, N * J)
    gram = (G.T @ G) / T
    rhs = G.mean(axis=0)
    gram.flat[:: gram.shape[0] + 1] += ridge
    try:
        cf = cho_factor(gram, lower=True, check_finite=False)
        coef = cho_solve(cf, rhs, check_finite=False)
    except np.linalg.LinAlgError:
        coef = np.linalg.solve(gram, rhs)
    return coef.reshape(N, J)


def evaluate_policy(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return features @ coef.T


def population_sharpe(weights: np.ndarray, lambda_z: np.ndarray,
                       econ: Economy = ECON) -> dict[str, float]:
    mu_z = lambda_z @ econ.B.T
    mean_return = float(np.mean(np.sum(weights * mu_z, axis=1)))
    second_moment = float(np.mean(np.einsum("ti,ij,tj->t", weights, econ.Omega, weights)))
    variance = max(second_moment - mean_return**2, 1e-14)
    sr_period = mean_return / math.sqrt(variance)
    sr_annual = math.sqrt(econ.periods_per_year) * sr_period
    sr2_annual = sr_annual**2
    oracle_sr2 = TARGET_ANNUAL_SR**2
    return {
        "mean_return": mean_return,
        "second_moment": second_moment,
        "sharpe_annual": sr_annual,
        "sharpe2_annual": sr2_annual,
        "sharpe2_recovery": sr2_annual / oracle_sr2,
        "sharpe2_shortfall": max(oracle_sr2 - sr2_annual, 1e-12),
    }


def evaluation_grid(m: int = 13) -> np.ndarray:
    sampler = qmc.Sobol(d=D, scramble=False)
    return sampler.random_base2(m=m)


Z_EVAL = evaluation_grid(13)
LAMBDA_LINEAR_EVAL = lambda_linear(Z_EVAL)
LAMBDA_NONLINEAR_EVAL = lambda_nonlinear(Z_EVAL)
TRUE_W_LINEAR_EVAL = true_policy(LAMBDA_LINEAR_EVAL)
TRUE_W_NONLINEAR_EVAL = true_policy(LAMBDA_NONLINEAR_EVAL)


def oracle_checks() -> dict[str, float]:
    max_lin = float(np.linalg.norm(LAMBDA_LINEAR_EVAL, axis=1).max())
    max_non = float(np.linalg.norm(LAMBDA_NONLINEAR_EVAL, axis=1).max())
    lin = population_sharpe(TRUE_W_LINEAR_EVAL, LAMBDA_LINEAR_EVAL)
    non = population_sharpe(TRUE_W_NONLINEAR_EVAL, LAMBDA_NONLINEAR_EVAL)
    return {
        "target_m": TARGET_M,
        "analytic_oracle_sr_annual": annual_sharpe_from_m(TARGET_M, ECON.periods_per_year),
        "quadrature_linear_sr_annual": lin["sharpe_annual"],
        "quadrature_nonlinear_sr_annual": non["sharpe_annual"],
        "max_norm_lambda_linear": max_lin,
        "max_norm_lambda_nonlinear": max_non,
    }


# ---------------------------------------------------------------------------
# Monte Carlo tasks
# ---------------------------------------------------------------------------
def nonlinear_feature_count(T: int, ridge_scale: float) -> int:
    ridge = ridge_scale * T ** (-THEORY_LAMBDA_EXP)
    effective = ridge ** (-D / (2.0 * S))
    # Number of real nonlinear features. Keep complete cosine/sine pairs.
    J = int(math.ceil(2.5 * effective))
    J = max(17, min(71, J))
    if J % 2 == 0:
        J += 1
    return J


def exp1_task(T: int, rep: int, seed: int,
              eval_features: dict[str, np.ndarray]) -> list[dict]:
    rng = np.random.default_rng(seed + 100_000 * T + rep)
    z = simulate_state(T, D, ECON.rho, rng)
    lambda_z = lambda_linear(z)
    R = sample_returns(lambda_z, rng)

    methods = {
        "Linear": legendre_features(z, 1),
        "Quadratic": legendre_features(z, 2),
        "Cubic": legendre_features(z, 3),
    }
    rows = []
    ridge = 0.25 / T
    for method, feat in methods.items():
        coef = fit_direct_policy(feat, R, ridge)
        w_eval = evaluate_policy(eval_features[method], coef)
        metrics = population_sharpe(w_eval, LAMBDA_LINEAR_EVAL)
        rows.append({"experiment": "linear", "T": T, "rep": rep, "method": method,
                     "ridge": ridge, "n_features": feat.shape[1], **metrics})
    return rows


def exp2_task(T: int, rep: int, seed: int, ridge_scale: float,
              modes: list[np.ndarray], eval_features: dict[str, np.ndarray]) -> list[dict]:
    rng = np.random.default_rng(seed + 300_000_000 + 100_000 * T + rep)
    z = simulate_state(T, D, ECON.rho, rng)
    lambda_z = lambda_nonlinear(z)
    R = sample_returns(lambda_z, rng)

    ridge_sob = ridge_scale * T ** (-THEORY_LAMBDA_EXP)
    train_features = {
        "Linear": legendre_features(z, 1),
        "Quadratic": legendre_features(z, 2),
        "Sobolev": flexible_features(z, modes, S),
    }
    ridges = {"Linear": 0.25 / T, "Quadratic": 0.25 / T, "Sobolev": ridge_sob}

    rows = []
    for method, feat in train_features.items():
        coef = fit_direct_policy(feat, R, ridges[method])
        w_eval = evaluate_policy(eval_features[method], coef)
        metrics = population_sharpe(w_eval, LAMBDA_NONLINEAR_EVAL)
        rows.append({"experiment": "nonlinear", "T": T, "rep": rep, "method": method,
                     "ridge": ridges[method], "n_features": feat.shape[1], **metrics})
    return rows


def exp2_sobolev_only_task(T: int, rep: int, seed: int, ridge_scale: float,
                            modes: list[np.ndarray], eval_sobolev: np.ndarray) -> dict:
    """Rate-extension task: fit only the Sobolev policy at a large sample size."""
    rng = np.random.default_rng(seed + 700_000_000 + 100_000 * T + rep)
    z = simulate_state(T, D, ECON.rho, rng)
    lambda_z = lambda_nonlinear(z)
    R = sample_returns(lambda_z, rng)
    features = flexible_features(z, modes, S)
    ridge = ridge_scale * T ** (-THEORY_LAMBDA_EXP)
    coef = fit_direct_policy(features, R, ridge)
    weights = evaluate_policy(eval_sobolev, coef)
    metrics = population_sharpe(weights, LAMBDA_NONLINEAR_EVAL)
    return {"experiment": "nonlinear_rate", "T": T, "rep": rep, "method": "Sobolev",
            "ridge": ridge, "n_features": features.shape[1], **metrics}
