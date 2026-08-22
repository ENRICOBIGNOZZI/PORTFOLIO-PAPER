#!/usr/bin/env python3
"""Post-process the symmetric Linear/Sobolev x {SNR,N,d} Monte Carlo."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import finance_monte_carlo as mc
import finance_monte_carlo_expanded as ex

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

SNR_LEVELS = (0.50, 1.00, 1.50)
N_LEVELS = (6, 20, 50)
D_LEVELS = (1, 2, 3)
THRESHOLDS = (0.75, 0.90, 0.95)

plt.rcParams.update({
    "font.size": 10.0,
    "axes.titlesize": 11.2,
    "axes.labelsize": 10.0,
    "legend.fontsize": 8.3,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def _ensure_relative_shortfall(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "relative_shortfall" not in out:
        out["relative_shortfall"] = np.maximum(1.0 - out["sharpe2_recovery"], 1e-12)
    if "sharpe2_recovery" not in out:
        out["sharpe2_recovery"] = out["sharpe2_annual"] / out["oracle_sr_annual"] ** 2
    return out


def load_all() -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []

    lin_snr = _ensure_relative_shortfall(pd.read_csv(RESULTS / "mc_symmetric_linear_snr_raw.csv"))
    lin_snr = lin_snr.assign(design="SNR", level=lin_snr["portfolio_snr"], level_label=lin_snr["oracle_sr_annual"].map(lambda x: f"SR*={x:.1f}"))
    blocks.append(lin_snr)

    sob_snr = _ensure_relative_shortfall(pd.read_csv(RESULTS / "mc_snr_raw.csv"))
    sob_snr = sob_snr.assign(
        model="Sobolev", design="SNR", level=sob_snr["portfolio_snr"],
        level_label=sob_snr["oracle_sr_annual"].map(lambda x: f"SR*={x:.1f}"),
        N=6, d=3, theory_rate=2.0 / 3.0,
    )
    blocks.append(sob_snr)

    lin_n = _ensure_relative_shortfall(pd.read_csv(RESULTS / "mc_symmetric_linear_N_raw.csv"))
    lin_n = lin_n.assign(design="N", level=lin_n["N"].astype(float), level_label=lin_n["N"].map(lambda x: f"N={int(x)}"))
    blocks.append(lin_n)

    sob_n = _ensure_relative_shortfall(pd.read_csv(RESULTS / "mc_asset_dimension_raw.csv"))
    sob_n = sob_n.assign(
        model="Sobolev", design="N", level=sob_n["N"].astype(float),
        level_label=sob_n["N"].map(lambda x: f"N={int(x)}"), d=3,
        theory_rate=2.0 / 3.0,
    )
    blocks.append(sob_n)

    dim = _ensure_relative_shortfall(pd.read_csv(RESULTS / "mc_symmetric_dimension_raw.csv"))
    dim = dim.assign(design="d", level=dim["d"].astype(float), level_label=dim["d"].map(lambda x: f"d={int(x)}"))
    blocks.append(dim)

    sob_d3 = _ensure_relative_shortfall(pd.read_csv(RESULTS / "mc_nonlinear_convergence_raw.csv"))
    sob_d3 = sob_d3[sob_d3["T"].isin(sorted(dim["T"].unique()))].copy()
    sob_d3 = sob_d3.assign(
        model="Sobolev", design="d", level=3.0, level_label="d=3",
        N=6, d=3, theory_rate=2.0 / 3.0,
    )
    blocks.append(sob_d3)

    keep = [
        "design", "model", "level", "level_label", "T", "rep", "N", "d",
        "oracle_sr_annual", "portfolio_snr", "theory_rate", "sharpe_annual",
        "sharpe2_annual", "sharpe2_recovery", "relative_shortfall",
    ]
    harmonized = []
    for block in blocks:
        for col in keep:
            if col not in block:
                block[col] = np.nan
        harmonized.append(block[keep])
    out = pd.concat(harmonized, ignore_index=True)
    return out.sort_values(["model", "design", "level", "T", "rep"]).reset_index(drop=True)

