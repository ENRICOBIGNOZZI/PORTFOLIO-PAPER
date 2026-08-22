#!/usr/bin/env python3
"""Symmetric Monte Carlo comparative statics for portfolio learning.

The paper-facing design is a 2 x 3 matrix:

    correctly specified linear policy   x {signal strength, N, state dimension d}
    correctly specified Sobolev policy  x {signal strength, N, state dimension d}

Only one primitive moves inside each panel.  The signal and asset-dimension
Sobolev cells already produced by the validated release are retained.  This
script generates the missing linear signal and N cells and the state-dimension
cells for both models.  Every sample-size cell is based on independent histories.

The common economic criterion is the annualized population Sharpe ratio, evaluated
by deterministic Sobol integration under the known DGP.  The reported relative
shortfall is 1 - SR(hat w)^2 / SR*^2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

import finance_monte_carlo as mc
import comparative_statics_mc as cs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SNR_LEVELS = (0.50, 1.00, 1.50)
N_LEVELS = (6, 20, 50)
D_LEVELS = (1, 2, 3)
BASE_SR = 1.50
BASE_RHO = 0.55
RIDGE_LINEAR_SCALE = 0.25
RIDGE_SOBOLEV_SCALE = 2.0

SNR_T = (800, 1250, 2000, 3200, 5000, 8000, 12000, 18000, 28000, 42000, 64000)
N_T = (800, 1250, 2000, 3200, 5000, 8000, 12000, 18000, 28000, 42000)
D_T = SNR_T


def target_m(sr_annual: float) -> float:
    return mc.target_m_from_annual_sharpe(sr_annual, mc.ECON.periods_per_year)


def portfolio_snr(sr_annual: float) -> float:
    return sr_annual**2 / mc.ECON.periods_per_year


def common_seed(seed: int, block: int, T: int, rep: int) -> int:
    return int(seed + block * 1_000_000_000 + T * 10_000 + rep)


def evaluation_grid(d: int, m: int = 13) -> np.ndarray:
    return qmc.Sobol(d=d, scramble=False).random_base2(m)


def factor_eigenvectors() -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(mc.ECON.H)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


H_EIGVALS, H_EIGVECS = factor_eigenvectors()


def linear_dimension_loadings(d: int, sr_annual: float = BASE_SR) -> np.ndarray:
    """K x d loading matrix with equal opportunity share per state coordinate.

    Columns are factor-space eigenvectors.  The scaling makes
    E[lambda(Z)' H lambda(Z)] equal to the target opportunity index exactly.
    With q_j(Z_j)=sqrt(3)(2Z_j-1), the q_j are orthonormal.
    """
    m = target_m(sr_annual)
    amplitudes = np.sqrt(m / (d * H_EIGVALS[:d]))
    return H_EIGVECS[:, :d] * amplitudes[None, :]


def lambda_linear_dimension(z: np.ndarray, d: int, sr_annual: float = BASE_SR) -> np.ndarray:
    q = mc.legendre_features(z, 1)[:, 1:]
    return q @ linear_dimension_loadings(d, sr_annual).T


def linear_dimension_oracle_check(d: int, sr_annual: float = BASE_SR) -> dict[str, float]:
    z = evaluation_grid(d, 15)
    lam = lambda_linear_dimension(z, d, sr_annual)
    weights = mc.true_policy(lam, mc.ECON)
    metrics = cs.population_metrics_econ(weights, lam, mc.ECON, sr_annual)
    return {
        "quadrature_oracle_sharpe": metrics["sharpe_annual"],
        "max_norm_lambda": float(np.linalg.norm(lam, axis=1).max()),
    }

