# JKP Common Task Framework: empirical portfolio-complexity figures

This folder implements the paper's four headline empirical complexity exercises on the **Jensen–Kelly–Pedersen Common Task Framework (CTF)** stock-level dataset used by the public leaderboard at <https://jkpfactors.com/ctf/leaderboard>.

The code does **not** redistribute CTF/WRDS data. Obtain the three CTF tables through WRDS following the official JKP guide, and place at least these two files locally:

- `data/raw/ctff_chars.parquet`
- `data/raw/ctff_features.parquet`

`ctff_daily_ret.parquet` is needed only for the standalone CTF submission contract, not for the monthly paper figures.

If you have WRDS access, `download_ctf_from_wrds.py` streams the three official tables directly from `contrib_global_factor` into local Parquet files without hard-coding credentials.

## Economic/statistical object

For stock `i` at month `t`, let `x_it` denote the contemporaneously observed JKP characteristics and `r_i,t+1` the provided one-month-ahead excess return. We estimate a scalar characteristic-to-weight policy

`w_it = f(x_it) / N_t`

by response-one ridge regression. For a feature map `phi`, define the ex-post monthly managed feature

`F_t = N_t^{-1} sum_i r_i,t+1 phi(x_it)`.

Then the portfolio estimator is exactly

`beta_hat = (F'F/T + lambda I)^(-1) F'1/T`,

and its realized portfolio return is `F_t beta_hat`. The effective complexity is computed from the **managed-payoff** Gram matrix,

`C(lambda) = tr[G (G + lambda I)^(-1)]`, with `G = F F'/T`,

rather than from the raw characteristic kernel alone.

## Primary design: one kernel, one changing object

The headline figures use **one Matérn-3/2 kernel**. A single large nested random-feature approximation is fixed, and only `lambda` changes. This isolates effective complexity from changes in functional form.

1. `fig1_oos_sharpe_vs_effective_complexity`: OOS Sharpe as a function of `C`, with the observed maximum marked. The code does not force an interior maximum.
2. `fig2_optimal_complexity_vs_sample_size`: `C*_T` against training history `T` on log-log axes.
3. `fig3_in_sample_vs_oos_sharpe`: the in-sample/OOS complexity wedge.
4. `fig4_nominal_vs_effective_complexity`: nested Matérn random-feature dimension `P` versus the Sharpe-maximizing effective complexity.
5. `fig5_kernel_robustness`: separate robustness panel (Matérn-1/2, Matérn-3/2, Matérn-5/2, RBF and linear).

The stationary-kernel length scale is the median pairwise distance of cross-sectional percentile-ranked characteristics computed **only from the pre-test sample**. Characteristic preprocessing at month `t` uses only that month's cross section.

## Run

Optional WRDS download:

```bash
cd empirical_jkp_ctf
python download_ctf_from_wrds.py --out data/raw
```

Then generate the paper figures:

```bash
python run_paper_figures.py \
  --chars data/raw/ctff_chars.parquet \
  --features data/raw/ctff_features.parquet \
  --out outputs/jkp_ctf
```

Defaults:

- primary training window: `T=120` months;
- `T` grid: `36,60,120,180,240`;
- maximum Matérn RFF dimension: `P=4096`;
- nominal-dimension grid: `32,64,128,256,512,1024,2048,4096`;
- Matérn smoothness: `nu=1.5`.

For a quick first pass, add `--skip-robustness`.

The script writes all figure PDFs/PNGs plus CSV audit trails and `run_metadata.json`. In particular, the metadata records whether the observed OOS Sharpe maximum is actually interior.

## CTF-compatible model

`ctf_submission_matern.py` provides the exact CTF

```python
def main(chars, features, daily_ret) -> pd.DataFrame
```

signature and returns `id, eom, w`. It uses a fixed, conservative Matérn-3/2 specification and only trailing observations. It is included as a reproducible bridge to the leaderboard; **no leaderboard performance claim is made until the model has actually been run/scored by the CTF pipeline**.

## Data citation

Use the JKP citation requested by the data provider:

Jensen, Theis Ingerslev, Bryan Kelly, and Lasse Heje Pedersen (2023), “Is There a Replication Crisis in Finance?”, *Journal of Finance* 78(5), 2465–2518.
