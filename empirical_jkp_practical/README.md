# Practical portfolio learnability on public JKP portfolios

This is an executed research study, not a CTF leaderboard submission. It allocates across **153 official Jensen-Kelly-Pedersen characteristic-managed long-short portfolios**. It does not estimate stock-level weights from the licensed CTF panel, certify previous project PDFs, or claim live profitability.

**Start with [the generated evidence report](results/REPORT.md) and [the PDF](results/JKP_Practical_Learnability.pdf).** Results and figures are generated from CSVs, not copied from earlier narratives.

## Meaning of complexity

`C = tr[G (G + lambda I)^(-1)]`, where `G_ts = K(Z_t,Z_s) R_t'R_s / T` is the managed-payoff Gram matrix. Only this effective dimension is called portfolio complexity. Feature count, state dimension and random-feature count describe a representation. Each training sample maps a prescribed C budget into lambda using its own eigenvalues. This empirical data-dependent procedure is not a structural estimator of b, r, or the theoretical oracle schedule.

The main exact Matérn-3/2 representation is fixed before validation. Other kernels are representation comparisons, not pure complexity effects. The constant-kernel benchmark has no state-dependent timing but is reestimated monthly: it is not a fixed buy-and-hold allocation.

## Locked chronology

- Data for this study: January 1973-December 2025. Calendar dates are return-realization months.
- Universe completeness, state normalization and median kernel length scale use data through December 1994 only.
- Validation: January 1995-December 2004.
- Test: January 2005-December 2025 (252 monthly observations).
- Monthly trailing training windows: 120 months; sensitivities at 60 and 240 months on the same evaluation dates.
- State: lagged 12-month mean and volatility of seven original-literature groups, 14 coordinates. These are **not** the 13 estimated JKP theme clusters.
- Current-month returns do not enter current weights. Official portfolio returns already include the literature orientation; do not multiply by the `direction` field again.

The full characteristic library is retrospective: some papers appeared after the test began. A 57-characteristic pre-2005 publication subset is separately evaluated. This does not reconstruct historical data vintages or eliminate every form of predictor-discovery bias.

## Experiments

The study reports full C paths, validation-selected models, static/linear/Gaussian alternatives, feature-library sizes, group and individual ablations, state ablations, training-history sensitivity, nested Matérn random-feature approximations, weighting schemes, spectral diagnostics and cost-proxy sensitivity.

Individual-feature rankings are computed on validation only. Top 8/16/32/64 libraries are then evaluated on the untouched test, after rebuilding states using only their retained features. The winning library size is also selected on validation. The best test library must not replace that selection.

Ablations separate removing an investment sleeve from removing a state predictor. Group and individual investment-sleeve comparisons report both retuned C and matched C. These are correlated-feature predictive comparisons, not causal effects or unique alpha attribution.

## Finance interpretation and limits

The deployed policy uses a causal 10% annualized risk target, 10% diagonal covariance shrinkage and a cap of three on the sum of absolute factor-sleeve weights. Realized volatility may be much lower or higher than 10%. C labels the trained response-one core; native loss is recorded before risk scaling. Means use 12 times the monthly average; volatility and Sharpe use the conventional square-root-of-12 scaling, not a serial-correlation-adjusted long-horizon variance.

The risk-return plot contains constant downscalings of the actual selected return streams, ending at each observed volatility. It does not extrapolate the high-Sharpe static portfolio to 10% risk while ignoring its binding exposure cap. These are estimated attainable directions, not a known population efficient frontier or measured oracle gap.

The cost stress subtracts 0/10/25/50 basis points times `sum(abs(w_t-w_(t-1)))`. This is only **factor-sleeve reallocation**. It omits stock-level internal factor turnover, netting, borrowing, spreads, price impact and capacity. It is neither full net trading performance nor necessarily a lower bound on actual costs.

The block bootstrap resamples paired selected-model returns, with block lengths 6/12/24 and 95% intervals. Family-wide bands are reported for group/state/top-10 ablations; the C-curve band is pointwise. These conditional intervals do not re-run feature/model selection. This is exploratory research, not confirmatory proof of a universal complexity law.

No b or r estimate, universal hump, monotone kernel ranking, RFF plateau, optimal live feature list, or stock-level capacity claim is imposed.

## Reproduce

Python 3.13; numerical versions in `requirements.txt`. Source URLs, timestamps, hashes and versions are saved in the manifest. For an exact historical reproduction use the archived raw snapshot: the live endpoint can later revise history.

```bash
python -m pip install -r empirical_jkp_practical/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s empirical_jkp_practical/tests -v
python empirical_jkp_practical/fetch_public.py --out jkp_public_raw
python empirical_jkp_practical/run_study.py --raw jkp_public_raw --out empirical_jkp_practical/results
python empirical_jkp_practical/build_report.py --out empirical_jkp_practical/results
```

The workflow `jkp_practical_study.yml` executes these steps, renders the PDF for preflight, publishes evidence only on this research branch, and uploads a replication artifact. No trading or account access is performed.

## Key outputs

`selected_models.csv`, `complexity_curves.csv`, `primary_path_OOS.csv`, `primary_path_validation.csv`, `feature_ranking_VALIDATION_ONLY.csv`, `VALIDATION_SELECTED_characteristics.csv`, `group_ablation.csv`, `state_ablation.csv`, `top10_feature_holdout_checks.csv`, `history_comparison.csv`, `cost_PROXY_sensitivity.csv`, `nominal_size_comparison.csv`, `selected_factor_weights.csv`, `oos_spectral_contributions.csv`, `paired_inference.csv`, `manifest.json`, and one CSV per selected return path.

## What remains for stock-level characteristics

The licensed CTF panel is required to repeat this experiment for the scalar stock-weight function rather than factor allocations. Define the investible universe and cross-sectional ranks before observing lead returns; never filter current holdings by next-month return availability. Use verified delisting/return treatment or fail explicitly on unresolved missing payoffs. Tune feature groups, kernels, history and C only before test; compute turnover from drifted underlying holdings, account for netting and borrowing, and run the same matched-C ablations.

The earlier `empirical_jkp_ctf` directory is not automatically certified by this public-factor study. In particular, a factor-level shortlist is not evidence that the same raw inputs are optimal in a nonlinear stock-weight function.

## Attribution / data license

Jensen, Theis Ingerslev, Bryan Kelly, and Lasse Heje Pedersen (2023), *Is There a Replication Crisis in Finance?*, Journal of Finance 78, 2465-2518. Official source: https://jkpfactors.com/data and its linked public data endpoint. Metadata: the official `bkelly-lab/jkp-data` repository. The public data are **CC BY-NC 4.0**; this artifact is for non-commercial research. No licensed stock-level data are redistributed.
