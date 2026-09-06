from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://jkpfactors-data.s3.amazonaws.com/public"
GRID = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
BASE = ["market_equity", "be_me", "ret_12_1"]

# Frozen before this global extension: this is exactly the 40-factor library used
# in the prior USA/GBR/JPN exercise. No new country can alter this list.
COMMON40 = [
    "market_equity", "be_me", "ret_12_1", "at_be", "at_me", "at_turnover",
    "beta_dimson_21d", "coskew_21d", "debt_me", "dgp_dsale", "div12m_me",
    "dsale_dinv", "dsale_drec", "dsale_dsga", "eq_dur", "f_score", "inv_gr1a",
    "ival_me", "ivol_capm_252d", "kz_index", "ni_be", "ni_me", "noa_at", "noa_gr1a",
    "o_score", "oaccruals_at", "ocf_me", "pi_nix", "prc", "prc_highprc_252d",
    "rd_me", "rd_sale", "ret_1_0", "ret_3_1", "ret_6_1", "ret_9_1", "sale_gr1",
    "sale_gr3", "sale_me", "z_score",
]

DEFAULT_LOCATIONS = [
    "world", "world_ex_us", "developed", "emerging", "frontier",
    "usa", "gbr", "jpn", "can", "aus", "deu", "fra", "che", "hkg", "nld",
    "swe", "ita", "esp", "sgp", "kor", "bra", "ind", "chn", "zaf",
]


@dataclass
class Fit:
    weights: np.ndarray
    c_total: np.ndarray
    cancellation: np.ndarray
    numerical_rank: int


def _ridge_for_c(eig: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(eig) == 0:
        return np.full(len(targets), np.inf), np.zeros(len(targets))
    rank = len(eig)
    wanted = np.minimum(np.asarray(targets, float), rank * (1.0 - 1e-7))
    top = float(eig.max())
    lo = np.full(len(wanted), np.log(top) - 35.0)
    hi = np.full(len(wanted), np.log(top) + 35.0)
    for _ in range(55):
        mid = (lo + hi) / 2.0
        lam = np.exp(mid)
        c = (eig[:, None] / (eig[:, None] + lam[None, :])).sum(0)
        lo = np.where(c > wanted, mid, lo)
        hi = np.where(c > wanted, hi, mid)
    return np.exp((lo + hi) / 2.0), wanted


def fit_gate(R: np.ndarray, n_base: int = 3) -> Fit:
    """Protected baseline + residual spectral ridge with absolute rank tolerance."""
    R = np.asarray(R, float)
    if R.ndim != 2 or not np.isfinite(R).all() or len(R) <= n_base:
        raise ValueError("invalid training matrix")
    B, X = R[:, :n_base], R[:, n_base:]
    T, d = R.shape
    if np.linalg.matrix_rank(B) != n_base:
        raise ValueError("rank-deficient baseline")

    a0 = np.linalg.lstsq(B, np.ones(T), rcond=None)[0]
    hedge = np.linalg.lstsq(B, X, rcond=None)[0]
    QB, _ = np.linalg.qr(B, mode="reduced")
    H = X - QB @ (QB.T @ X)
    S = (H.T @ H) / T
    eig, V = np.linalg.eigh((S + S.T) / 2.0)

    # Backward-error floor tied to the original payoff scale rather than to the
    # residual itself. Exact redundancy therefore cannot be promoted by roundoff.
    scale = max(float(np.linalg.norm(X, 2)), float(np.linalg.norm(B, 2)), 1e-300)
    singular_floor = np.finfo(float).eps * max(T, X.shape[1], n_base) * scale
    eigen_floor = singular_floor**2 / T
    keep = eig > eigen_floor
    eig = eig[keep]
    V = V[:, keep]

    W = np.zeros((d, len(GRID)))
    W[:n_base] = a0[:, None]
    c_new = np.zeros(len(GRID))
    use = np.flatnonzero(GRID > 0)
    if len(eig) and len(use):
        lam, actual = _ridge_for_c(eig, GRID[use])
        mu = H.mean(0)
        b = V @ ((V.T @ mu)[:, None] / (eig[:, None] + lam[None, :]))
        W[n_base:, use] = b
        W[:n_base, use] -= hedge @ b
        c_new[use] = actual

    cancel = np.ones(len(GRID))
    for j in use:
        b = W[n_base:, j]
        da = W[:n_base, j] - a0
        old_leg = B @ da
        new_leg = X @ b
        net = old_leg + new_leg
        cancel[j] = (np.linalg.norm(old_leg) + np.linalg.norm(new_leg)) / max(np.linalg.norm(net), 1e-30)

    return Fit(W, n_base + c_new, cancel, len(eig))


def sr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(12.0) * x.mean() / max(x.std(ddof=1), 1e-14))


def qloss(x: np.ndarray, axis=None) -> np.ndarray:
    return np.mean((1.0 - np.asarray(x, float)) ** 2, axis=axis)


def slug(location: str, weighting: str) -> str:
    return f"[{location}]_[all_factors]_[monthly]_[{weighting}]"


def download(raw: Path, location: str, weighting: str = "vw_cap") -> Path | None:
    raw.mkdir(parents=True, exist_ok=True)
    s = slug(location, weighting)
    path = raw / f"{s}.zip"
    if path.exists() and path.stat().st_size > 100:
        return path
    url = BASE_URL + "/" + s.replace("[", "%5B").replace("]", "%5D") + ".zip"
    try:
        r = requests.get(url, timeout=(10, 90))
        if r.status_code != 200 or len(r.content) < 100:
            print("UNAVAILABLE", location, weighting, r.status_code, flush=True)
            return None
        path.write_bytes(r.content)
        print("DOWNLOADED", location, weighting, len(r.content), flush=True)
        return path
    except Exception as exc:
        print("DOWNLOAD FAILED", location, weighting, repr(exc), flush=True)
        return None


def load_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"unexpected archive {path}")
        d = pd.read_csv(io.BytesIO(z.read(names[0])), parse_dates=["date"])
    p = d.pivot(index="date", columns="name", values="ret").sort_index()
    p.index = p.index.to_period("M").to_timestamp("M")
    return p


def contiguous_suffix(P: pd.DataFrame) -> pd.DataFrame:
    """Use the final contiguous interval on which the frozen library is observable."""
    valid = P.notna().all(axis=1)
    if len(valid) == 0 or not bool(valid.iloc[-1]):
        return P.iloc[0:0]
    bad = np.flatnonzero(~valid.to_numpy())
    start = int(bad[-1] + 1) if len(bad) else 0
    return P.iloc[start:]


def make_path(P: pd.DataFrame, T: int) -> dict:
    if len(P) < max(96, T + 120 + 36):
        raise ValueError("insufficient history")
    # A fixed ex-ante scale: the first eight years of the usable interval.
    rms = np.sqrt((P.iloc[:96] ** 2).mean(0)).replace(0, np.nan)
    if rms.isna().any() or (rms < 1e-12).any():
        raise ValueError("degenerate RMS scale")
    R = (P / rms).to_numpy(float)
    idx = np.arange(T, len(P))
    rr = np.empty((len(idx), len(GRID)))
    cc = np.empty_like(rr)
    cancellation = np.empty_like(rr)
    ranks = np.empty(len(idx), dtype=int)
    for n, t in enumerate(idx):
        fit = fit_gate(R[t - T:t], 3)
        rr[n] = R[t] @ fit.weights
        cc[n] = fit.c_total
        cancellation[n] = fit.cancellation
        ranks[n] = fit.numerical_rank
    return {
        "dates": P.index[idx],
        "returns": rr,
        "complexity": cc,
        "cancellation": cancellation,
        "rank": ranks,
    }


def masks(dd: pd.DatetimeIndex, year: int):
    old = (dd >= pd.Timestamp(year - 10, 1, 1)) & (dd < pd.Timestamp(year - 5, 1, 1))
    recent = (dd >= pd.Timestamp(year - 5, 1, 1)) & (dd < pd.Timestamp(year, 1, 1))
    allp = old | recent
    outer = (dd >= pd.Timestamp(year, 1, 1)) & (dd < pd.Timestamp(year + 1, 1, 1))
    return old, recent, allp, outer


def stack2_weight(p_gate: np.ndarray, p_all: np.ndarray) -> float:
    d = p_gate - p_all
    den = float(np.mean(d * d))
    if den < 1e-16:
        return 1.0
    return float(np.clip(np.mean(d * (1.0 - p_all)) / den, 0.0, 1.0))


def simplex_stack(P: np.ndarray) -> np.ndarray:
    """Exact active-set solution for <=3 nonnegative weights summing to one."""
    P = np.asarray(P, float)
    m = P.shape[1]
    if m == 1:
        return np.array([1.0])
    if m == 2:
        w = stack2_weight(P[:, 0], P[:, 1])
        return np.array([w, 1.0 - w])
    A = P.T @ P
    b = P.T @ np.ones(len(P))
    candidates: list[np.ndarray] = []
    K = np.block([[A, np.ones((m, 1))], [np.ones((1, m)), np.zeros((1, 1))]])
    try:
        sol = np.linalg.solve(K, np.r_[b, 1.0])[:m]
        if np.all(sol >= -1e-10):
            sol = np.maximum(sol, 0.0)
            candidates.append(sol / sol.sum())
    except np.linalg.LinAlgError:
        pass
    for i in range(m):
        for j in range(i + 1, m):
            w = stack2_weight(P[:, i], P[:, j])
            z = np.zeros(m); z[i] = w; z[j] = 1.0 - w
            candidates.append(z)
    for i in range(m):
        z = np.zeros(m); z[i] = 1.0
        candidates.append(z)
    losses = [float(np.mean((1.0 - P @ z) ** 2)) for z in candidates]
    return candidates[int(np.argmin(losses))]


def evaluate(dataset: str, location: str, weighting: str, T: int, path: dict) -> tuple[list[dict], list[dict]]:
    dd = path["dates"]
    rr = path["returns"]
    cc = path["complexity"]
    cancellation = path["cancellation"]
    selectors = ["current", "gate2", "allpast", "recent", "median3", "minimax", "mean3", "stack2", "stack3"]
    out = {k: [] for k in selectors}
    C = {k: [] for k in selectors}
    Cancel = {k: [] for k in selectors}
    base = []
    annual = []

    years = []
    for year in range(2005, 2026):
        old, recent, allp, outer = masks(dd, year)
        if old.sum() == 60 and recent.sum() == 60 and outer.sum() == 12:
            years.append(year)
    if len(years) < 3:
        return [], []

    for year in years:
        old, recent, allp, outer = masks(dd, year)
        lo = qloss(rr[old], axis=0)
        lr = qloss(rr[recent], axis=0)
        la = qloss(rr[allp], axis=0)
        jo, jr, ja = int(np.argmin(lo)), int(np.argmin(lr)), int(np.argmin(la))
        jg = min(jo, ja)
        jm = int(np.median([jo, jr, ja]))
        jw = int(np.argmin(np.maximum(lo, lr)))
        simple = {"current": jo, "gate2": jg, "allpast": ja, "recent": jr, "median3": jm, "minimax": jw}
        for name, j in simple.items():
            out[name].extend(rr[outer, j]); C[name].extend(cc[outer, j]); Cancel[name].extend(cancellation[outer, j])

        js3 = [jo, jr, ja]
        out["mean3"].extend(rr[outer][:, js3].mean(1))
        C["mean3"].extend(cc[outer][:, js3].mean(1))
        Cancel["mean3"].extend(cancellation[outer][:, js3].mean(1))

        wg = stack2_weight(rr[allp, jg], rr[allp, ja])
        out["stack2"].extend(wg * rr[outer, jg] + (1.0 - wg) * rr[outer, ja])
        C["stack2"].extend(wg * cc[outer, jg] + (1.0 - wg) * cc[outer, ja])
        Cancel["stack2"].extend(wg * cancellation[outer, jg] + (1.0 - wg) * cancellation[outer, ja])

        unique_js = list(dict.fromkeys([jo, jr, ja]))
        w3 = simplex_stack(rr[allp][:, unique_js])
        out["stack3"].extend(rr[outer][:, unique_js] @ w3)
        C["stack3"].extend(cc[outer][:, unique_js] @ w3)
        Cancel["stack3"].extend(cancellation[outer][:, unique_js] @ w3)

        base.extend(rr[outer, 0])
        annual.append({
            "dataset": dataset, "location": location, "weighting": weighting, "T": T, "year": year,
            "C_old": float(GRID[jo]), "C_recent": float(GRID[jr]), "C_all": float(GRID[ja]),
            "C_gate2": float(GRID[jg]), "C_median": float(GRID[jm]), "C_minimax": float(GRID[jw]),
            "stack2_w_gate": float(wg), "stack3_n_arms": len(unique_js),
            "selector_disagreement": float(max(jo, jr, ja) - min(jo, jr, ja)),
        })

    base = np.asarray(base)
    rows = []
    for name in selectors:
        x = np.asarray(out[name]); c = np.asarray(C[name]); z = np.asarray(Cancel[name])
        yearly_positive = []
        for i in range(len(years)):
            sl = slice(12 * i, 12 * (i + 1))
            yearly_positive.append(np.mean((1.0 - base[sl]) ** 2 - (1.0 - x[sl]) ** 2) > 0)
        rows.append({
            "dataset": dataset, "location": location, "weighting": weighting, "T": T,
            "start_year": years[0], "end_year": years[-1], "n_years": len(years), "n_months": len(x),
            "selector": name, "base_sr": sr(base), "sr": sr(x), "delta_sr": sr(x) - sr(base),
            "gainQ": float(np.mean((1.0 - base) ** 2 - (1.0 - x) ** 2)),
            "meanC": float(c.mean()), "sd_monthly_C": float(c.std(ddof=1)),
            "mean_cancellation": float(np.nanmean(z)), "p95_cancellation": float(np.nanquantile(z, 0.95)),
            "positive_year_frac": float(np.mean(yearly_positive)),
        })
    return rows, annual


def make_report(results: pd.DataFrame, annual: pd.DataFrame, skipped: list[dict]) -> str:
    if results.empty:
        return "# Global Gate stress test\n\nNo eligible datasets."
    agg = results.groupby("selector").agg(
        comparisons=("gainQ", "size"), datasets=("dataset", "nunique"),
        avg_gainQ=("gainQ", "mean"), median_gainQ=("gainQ", "median"),
        avg_sr=("sr", "mean"), avg_delta_sr=("delta_sr", "mean"),
        avg_C=("meanC", "mean"), avg_cancel=("mean_cancellation", "mean"),
        positive_year_frac=("positive_year_frac", "mean"),
    ).sort_values("avg_gainQ", ascending=False)
    current = results[results.selector == "current"][["dataset", "T", "gainQ", "sr"]].rename(columns={"gainQ": "g0", "sr": "s0"})
    comp = results.merge(current, on=["dataset", "T"], how="left")
    wins = []
    for selector, g in comp.groupby("selector"):
        wins.append({"selector": selector,
                     "q_wins": int((g.gainQ > g.g0 + 1e-12).sum()),
                     "q_losses": int((g.gainQ < g.g0 - 1e-12).sum()),
                     "sr_wins": int((g.sr > g.s0 + 1e-12).sum()),
                     "sr_losses": int((g.sr < g.s0 - 1e-12).sum())})
    wins = pd.DataFrame(wins).set_index("selector")
    lines = [
        "# Global Learnability Gate stress test", "",
        "The 40-factor library was frozen before the additional countries were downloaded. No new-country return is used to choose the factor set.", "",
        f"Eligible comparisons: **{len(current)}** across **{results.dataset.nunique()} datasets**.", "",
        "## Aggregate selector results", "", "```", agg.round(4).to_string(), "```", "",
        "## Wins and losses versus the original selector", "", "```", wins.to_string(), "```", "",
        "## Selectors", "",
        "- current: first five-year tuning block;",
        "- gate2: min(first-five-year choice, all-past choice);",
        "- allpast: all ten predeployment years;",
        "- recent: most recent five years;",
        "- median3: median complexity among old, recent and all-past choices;",
        "- minimax: minimizes the worse loss across the two five-year blocks;",
        "- mean3: equal-weight portfolio blend of old/recent/all-past choices;",
        "- stack2: response-one stacking of Gate 2.0 and all-past;",
        "- stack3: nonnegative response-one stacking of distinct old/recent/all-past policies.", "",
        "## Skipped / insufficient-history datasets", "", "```",
        pd.DataFrame(skipped).to_string(index=False) if skipped else "None", "```", "",
    ]
    if not annual.empty:
        lines += [
            "## Selector disagreement", "",
            f"Mean max-minus-min grid-index disagreement across annual old/recent/all choices: {annual.selector_disagreement.mean():.2f}.",
            f"Stack2 mean weight on conservative Gate 2.0: {annual.stack2_w_gate.mean():.3f}.", "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--locations", nargs="*", default=DEFAULT_LOCATIONS)
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = []
    for loc in args.locations:
        p = download(args.raw, loc, "vw_cap")
        if p is not None:
            datasets.append((f"{loc}_vwcap", loc, "vw_cap", p))
    for wt in ["vw", "ew"]:
        p = download(args.raw, "usa", wt)
        if p is not None:
            datasets.append((f"usa_{wt}", "usa", wt, p))

    rows: list[dict] = []
    annual: list[dict] = []
    skipped: list[dict] = []
    manifest: list[dict] = []

    for name, loc, wt, path in datasets:
        try:
            p = load_zip(path)
            missing = [c for c in COMMON40 if c not in p.columns]
            if missing:
                skipped.append({"dataset": name, "reason": f"missing {len(missing)} frozen factors"})
                continue
            usable = contiguous_suffix(p[COMMON40])
            if len(usable) < 240:
                skipped.append({"dataset": name, "reason": f"usable suffix only {len(usable)} months"})
                continue
            manifest.append({"dataset": name, "first": usable.index.min().date(), "last": usable.index.max().date(), "months": len(usable)})
            for T in [60, 120, 240]:
                try:
                    path_data = make_path(usable, T)
                    rr, aa = evaluate(name, loc, wt, T, path_data)
                    if not rr:
                        skipped.append({"dataset": f"{name}_T{T}", "reason": "fewer than 3 eligible deployment years"})
                        continue
                    rows.extend(rr); annual.extend(aa)
                except Exception as exc:
                    skipped.append({"dataset": f"{name}_T{T}", "reason": repr(exc)})
            print("DONE", name, len(usable), flush=True)
        except Exception as exc:
            skipped.append({"dataset": name, "reason": repr(exc)})
            print("FAILED", name, repr(exc), flush=True)

    res = pd.DataFrame(rows)
    ann = pd.DataFrame(annual)
    res.to_csv(args.out / "global_selector_results.csv", index=False)
    ann.to_csv(args.out / "annual_choices.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out / "skipped.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.out / "dataset_manifest.csv", index=False)
    pd.DataFrame({"factor": COMMON40, "role": ["baseline" if x in BASE else "extension" for x in COMMON40]}).to_csv(args.out / "frozen_common40.csv", index=False)
    if not res.empty:
        agg = res.groupby("selector").agg(
            comparisons=("gainQ", "size"), datasets=("dataset", "nunique"),
            avg_gainQ=("gainQ", "mean"), median_gainQ=("gainQ", "median"),
            avg_sr=("sr", "mean"), avg_delta_sr=("delta_sr", "mean"),
            avg_C=("meanC", "mean"), avg_cancel=("mean_cancellation", "mean"),
            positive_year_frac=("positive_year_frac", "mean"),
        ).reset_index()
        agg.to_csv(args.out / "selector_aggregate.csv", index=False)
    (args.out / "REPORT.md").write_text(make_report(res, ann, skipped), encoding="utf-8")
    print((args.out / "REPORT.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
