#!/usr/bin/env python3
"""Comparative statics for the Learning Portfolio Decisions Monte Carlo.

This module adds two controlled exercises to the nonlinear Sobolev economy:

1. Asset dimension N in {6,20,50}.  The economies are constructed so that
   B_N' D_N^{-1} B_N is identical across N.  Hence the factor-space opportunity
   matrix, the population maximum Sharpe ratio, the conditional factor-premium
   function, and the Sobolev/source geometry are exactly the same.  Only the
   dimension of the unrestricted portfolio-weight vector changes.

2. State persistence rho in {0,0.55,0.85}.  The stationary marginal law of Z_t
   remains uniform for every rho, so the population target and economic spectrum
   are unchanged.  Only temporal dependence changes.

Every (design,T,replication) cell uses a newly generated market history.  Policy
performance is evaluated at the population level by the same deterministic Sobol
quadrature used in the main Monte Carlo.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import lfilter

import finance_monte_carlo as mc
import finance_monte_carlo_expanded as ex

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

N_LEVELS = (6, 20, 50)
RHO_LEVELS = (0.00, 0.55, 0.85, 0.95)


def symmetric_sqrt(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    if float(values.min()) <= 0:
        raise ValueError("matrix is not positive definite")
    return (vectors * np.sqrt(values)[None, :]) @ vectors.T


def symmetric_inv_sqrt(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    if float(values.min()) <= 0:
        raise ValueError("matrix is not positive definite")
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def base_factor_information() -> np.ndarray:
    d0 = np.diag(mc.ECON.idio_sd**2)
    return mc.ECON.B.T @ np.linalg.solve(d0, mc.ECON.B)


S_FACTOR = base_factor_information()
S_FACTOR_SQRT = symmetric_sqrt(S_FACTOR)


def economy_for_n(n_assets: int) -> mc.Economy:
    if n_assets == mc.ECON.N:
        return mc.ECON
    if n_assets < mc.ECON.K:
        raise ValueError("N must be at least the number of factors")

    x = (np.arange(n_assets, dtype=float) + 0.5) / n_assets
    raw = np.column_stack([
        0.80 + 0.30 * np.cos(2 * np.pi * x) + 0.12 * np.sin(6 * np.pi * x),
        0.18 + 0.52 * np.sin(2 * np.pi * x) + 0.24 * np.cos(4 * np.pi * x),
        0.10 + 0.46 * np.cos(2 * np.pi * x) - 0.32 * np.sin(4 * np.pi * x),
    ])
    idio_sd = 0.46 + 0.10 * (0.5 + 0.5 * np.sin(2 * np.pi * x + 0.35))
    d_inv_raw = raw / (idio_sd[:, None] ** 2)
    raw_information = raw.T @ d_inv_raw
    transform = symmetric_inv_sqrt(raw_information) @ S_FACTOR_SQRT
    b = raw @ transform
    econ = mc.Economy(B=b, idio_sd=idio_sd, rho=mc.ECON.rho,
                      periods_per_year=mc.ECON.periods_per_year)
    discrepancy = np.max(np.abs(econ.H - mc.ECON.H))
    if discrepancy > 1e-10:
        raise RuntimeError(f"factor-opportunity normalization failed for N={n_assets}: {discrepancy}")
    return econ


ECONOMIES = {n: economy_for_n(n) for n in N_LEVELS}


def simulate_state_fast(t: int, d: int, rho: float, rng: np.random.Generator,
                        burn: int = 300) -> np.ndarray:
    n = t + burn
    initial = rng.standard_normal(d)
    innovations = rng.standard_normal((n, d))
    scale = math.sqrt(1.0 - rho**2)
    latent, _ = lfilter([scale], [1.0, -rho], innovations, axis=0,
                        zi=(rho * initial)[None, :])
    return mc.ndtr(latent[burn:])


def sample_returns_with_common_shocks(lambda_z: np.ndarray, factor_shocks: np.ndarray,
                                      idio_shocks: np.ndarray, econ: mc.Economy) -> np.ndarray:
    norms = np.linalg.norm(lambda_z, axis=1)
    if float(norms.max()) >= 0.96:
        raise ValueError("factor premia violate I-lambda lambda' >= 0")
    f = factor_shocks.copy()
    nz = norms > 1e-14
    if np.any(nz):
        unit = lambda_z[nz] / norms[nz, None]
        projection = np.sum(unit * factor_shocks[nz], axis=1)
        adjustment = np.sqrt(1.0 - norms[nz] ** 2) - 1.0
        f[nz] += (adjustment * projection)[:, None] * unit
    f += lambda_z
    return f @ econ.B.T + idio_shocks[:, : econ.N] * econ.idio_sd[None, :]


def population_metrics_econ(weights: np.ndarray, lambda_eval: np.ndarray,
                           econ: mc.Economy,
                           oracle_sr_annual: float = mc.TARGET_ANNUAL_SR) -> dict[str, float]:
    mu_z = lambda_eval @ econ.B.T
    mean_return = float(np.mean(np.sum(weights * mu_z, axis=1)))
    second_moment = float(np.mean(np.einsum("ti,ij,tj->t", weights, econ.Omega, weights)))
    variance = max(second_moment - mean_return**2, 1e-14)
    sr_annual = math.sqrt(econ.periods_per_year) * mean_return / math.sqrt(variance)
    sr2 = sr_annual**2
    oracle_sr2 = oracle_sr_annual**2
    recovery = sr2 / oracle_sr2
    return {"mean_return": mean_return, "second_moment": second_moment,
            "sharpe_annual": sr_annual, "sharpe2_annual": sr2,
            "sharpe2_recovery": recovery,
            "relative_shortfall": max(1.0 - recovery, 1e-12),
            "absolute_shortfall": max(oracle_sr2 - sr2, 1e-12)}


def _kronecker_preconditioner(features: np.ndarray, econ: mc.Economy,
                              ridge: float) -> LinearOperator:
    n = econ.N
    j = features.shape[1]
    s = (features.T @ features) / features.shape[0]
    ov, ou = np.linalg.eigh(econ.Omega)
    sv, su = np.linalg.eigh(s)
    denom = ov[:, None] * sv[None, :] + ridge
    def matvec(x: np.ndarray) -> np.ndarray:
        c = x.reshape(n, j)
        rotated = ou.T @ c @ su
        solved = rotated / denom
        return (ou @ solved @ su.T).reshape(-1)
    return LinearOperator((n * j, n * j), matvec=matvec, dtype=float)


def fit_direct_policy_iterative(features: np.ndarray, returns: np.ndarray,
                                ridge: float, econ: mc.Economy,
                                rtol: float = 2e-7, maxiter: int = 160):
    t, j = features.shape
    n = returns.shape[1]
    rhs_matrix = (returns.T @ features) / t
    rhs = rhs_matrix.reshape(-1)
    def matvec(x: np.ndarray) -> np.ndarray:
        c = x.reshape(n, j)
        fitted = np.sum((returns @ c) * features, axis=1)
        out = (returns.T @ (fitted[:, None] * features)) / t
        out += ridge * c
        return out.reshape(-1)
    operator = LinearOperator((n * j, n * j), matvec=matvec, dtype=float)
    preconditioner = _kronecker_preconditioner(features, econ, ridge)
    iterations = 0
    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1
    x0 = preconditioner @ rhs
    solution, info = cg(operator, rhs, x0=x0, M=preconditioner,
                        rtol=rtol, atol=0.0, maxiter=maxiter, callback=callback)
    residual = float(np.linalg.norm(operator @ solution - rhs) / max(np.linalg.norm(rhs), 1e-14))
    if info < 0:
        raise RuntimeError(f"CG failed with illegal input, info={info}")
    if info > 0 and residual > 2e-5:
        raise RuntimeError(f"CG failed to converge: info={info}, residual={residual:.3e}")
    return solution.reshape(n, j), iterations, residual


def validate_iterative_solver(seed: int = 20260818) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    t = 700
    z = mc.simulate_state(t, mc.D, mc.ECON.rho, rng)
    lam = mc.lambda_nonlinear(z)
    r = mc.sample_returns(lam, rng)
    modes = mc.fourier_modes(mc.D, 8)
    phi = mc.flexible_features(z, modes, mc.S)
    ridge = 2.0 * t ** (-mc.THEORY_LAMBDA_EXP)
    exact = mc.fit_direct_policy(phi, r, ridge)
    iterative, iters, residual = fit_direct_policy_iterative(phi, r, ridge, mc.ECON)
    relative_coef_error = float(np.linalg.norm(iterative - exact) / np.linalg.norm(exact))
    return {"relative_coefficient_error": relative_coef_error,
            "cg_iterations": iters,
            "relative_normal_equation_residual": residual}


def n_task(t: int, rep: int, seed: int, n_assets: int, max_n: int) -> dict:
    rng = np.random.default_rng(ex.seed_for(seed, 61, t, rep))
    z = mc.simulate_state(t, mc.D, mc.ECON.rho, rng)
    lam = mc.lambda_nonlinear(z)
    factor_shocks = rng.standard_normal((t, mc.ECON.K))
    idio_shocks = rng.standard_normal((t, max_n))
    econ = ECONOMIES[n_assets]
    returns = sample_returns_with_common_shocks(lam, factor_shocks, idio_shocks, econ)
    j_nonlin = mc.nonlinear_feature_count(t, 2.0)
    modes = mc.fourier_modes(mc.D, (j_nonlin - 1) // 2)
    features = mc.flexible_features(z, modes, mc.S)
    ridge = 2.0 * t ** (-mc.THEORY_LAMBDA_EXP)
    coef, iterations, residual = fit_direct_policy_iterative(features, returns, ridge, econ)
    eval_features = mc.flexible_features(mc.Z_EVAL, modes, mc.S)
    weights = mc.evaluate_policy(eval_features, coef)
    metrics = population_metrics_econ(weights, mc.LAMBDA_NONLINEAR_EVAL, econ)
    return {"experiment":"asset_dimension","N":n_assets,"T":t,"rep":rep,
            "method":"Sobolev","ridge_scale":2.0,"ridge":ridge,
            "n_features":features.shape[1],"n_coefficients":n_assets*features.shape[1],
            "oracle_sr_annual":mc.TARGET_ANNUAL_SR,
            "portfolio_snr":ex.portfolio_snr(mc.TARGET_ANNUAL_SR),
            "cg_iterations":iterations,"cg_residual":residual,**metrics}


def rho_task(t: int, rep: int, seed: int, rho: float, rho_id: int) -> dict:
    rng = np.random.default_rng(ex.seed_for(seed, 71, t, rep))
    z = simulate_state_fast(t, mc.D, rho, rng)
    lam = mc.lambda_nonlinear(z)
    returns = mc.sample_returns(lam, rng, mc.ECON)
    j_nonlin = mc.nonlinear_feature_count(t, 2.0)
    modes = mc.fourier_modes(mc.D, (j_nonlin - 1) // 2)
    features = mc.flexible_features(z, modes, mc.S)
    ridge = 2.0 * t ** (-mc.THEORY_LAMBDA_EXP)
    coef, iterations, residual = fit_direct_policy_iterative(features, returns, ridge, mc.ECON)
    weights = mc.evaluate_policy(mc.flexible_features(mc.Z_EVAL, modes, mc.S), coef)
    metrics = population_metrics_econ(weights, mc.LAMBDA_NONLINEAR_EVAL, mc.ECON)
    return {"experiment":"state_persistence","rho":rho,"T":t,"rep":rep,
            "method":"Sobolev","ridge_scale":2.0,"ridge":ridge,
            "n_features":features.shape[1],"oracle_sr_annual":mc.TARGET_ANNUAL_SR,
            "portfolio_snr":ex.portfolio_snr(mc.TARGET_ANNUAL_SR),
            "cg_iterations":iterations,"cg_residual":residual,**metrics}


def run_n_experiment(t_grid:list[int],reps:int,seed:int,jobs:int)->pd.DataFrame:
    rows=[]; max_n=max(N_LEVELS)
    for t in t_grid:
        rows.extend(Parallel(n_jobs=jobs,backend="threading",verbose=0)(
            delayed(n_task)(t,rep,seed,n,max_n) for rep in range(reps) for n in N_LEVELS))
    return pd.DataFrame(rows)

def run_rho_experiment(t_grid:list[int],reps:int,seed:int,jobs:int)->pd.DataFrame:
    rows=[]
    for t in t_grid:
        rows.extend(Parallel(n_jobs=jobs,backend="threading",verbose=0)(
            delayed(rho_task)(t,rep,seed,rho,rho_id) for rep in range(reps) for rho_id,rho in enumerate(RHO_LEVELS)))
    return pd.DataFrame(rows)

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--seed",type=int,default=20260818); parser.add_argument("--jobs",type=int,default=3); parser.add_argument("--quick",action="store_true"); args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    validation=validate_iterative_solver(args.seed+17)
    if validation["relative_coefficient_error"]>5e-5: raise RuntimeError(f"iterative solver validation failed: {validation}")
    if args.quick:
        n_protocol={800:2,2000:2}; rho_protocol={800:2,2000:2}
    else:
        n_protocol={800:48,1250:48,2000:48,3200:48,5000:48,8000:48,12000:48,18000:48,28000:24,42000:24,64000:24,80000:16,96000:16}
        rho_protocol={800:30,1250:30,2000:30,3200:30,5000:30,8000:30,12000:30,18000:30,28000:30,42000:30,64000:30}
    n_df=pd.concat([run_n_experiment([t],reps,args.seed+800_000,1) for t,reps in n_protocol.items()],ignore_index=True).sort_values(["T","rep","N"])
    rho_df=pd.concat([run_rho_experiment([t],reps,args.seed+900_000,args.jobs) for t,reps in rho_protocol.items()],ignore_index=True).sort_values(["T","rep","rho"])
    n_df.to_csv(RESULTS/"mc_asset_dimension_raw.csv",index=False); rho_df.to_csv(RESULTS/"mc_persistence_raw.csv",index=False)
    information_checks={}
    for n,econ in ECONOMIES.items():
        d=np.diag(econ.idio_sd**2); factor_information=econ.B.T@np.linalg.solve(d,econ.B)
        oracle_w=mc.true_policy(mc.LAMBDA_NONLINEAR_EVAL,econ); oracle=population_metrics_econ(oracle_w,mc.LAMBDA_NONLINEAR_EVAL,econ)
        information_checks[str(n)]={"max_abs_BDinvB_difference":float(np.max(np.abs(factor_information-S_FACTOR))),"max_abs_H_difference":float(np.max(np.abs(econ.H-mc.ECON.H))),"quadrature_oracle_sharpe":oracle["sharpe_annual"],"min_idiosyncratic_sd":float(econ.idio_sd.min()),"max_idiosyncratic_sd":float(econ.idio_sd.max())}
    metadata={"seed":args.seed,"asset_dimension_levels":list(N_LEVELS),"asset_dimension_replications_by_T":{str(t):r for t,r in n_protocol.items()},"persistence_levels":list(RHO_LEVELS),"persistence_replications_by_T":{str(t):r for t,r in rho_protocol.items()},"common_population_annual_sharpe":mc.TARGET_ANNUAL_SR,"common_sobolev_s":mc.S,"common_source_r":mc.R_SOURCE,"common_theoretical_exponent":mc.THEORY_RATE,"common_ridge_multiplier":2.0,"iterative_solver_validation":validation,"asset_economy_normalization_checks":information_checks}
    (RESULTS/"mc_comparative_statics_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
if __name__=="__main__": main()
