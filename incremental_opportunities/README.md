# Incremental opportunity learning

**Question:** what does a new characteristic add beyond an existing portfolio, and how much of that additional value can the available history recover?

This study replaces rankings of isolated strategies with nested policy extensions. The initial protocol was committed before the new calculations. The market-inclusive baseline is an explicitly secondary diagnostic, added after observing that the first three-tilt baseline performed poorly. Neither the earlier practical-study rankings nor its performance numbers enter these estimators.

## What was executed

The complete workflow passed: https://github.com/ENRICOBIGNOZZI/PORTFOLIO-PAPER/actions/runs/33966196416

- 16 tests: exact baseline nesting, profiled normal equations, projection orthogonality, smoother trace, rank handling, IS loss monotonicity, deterministic decisions, no-future-input recommendation, and exact population/spanning identities.
- Chronological JKP experiments with separate tuning, confirmation, and outer evaluation dates.
- 400 paired simulations per training history, with exact known population evaluation.
- All selected return series reconciled to their reported Sharpe ratios. A second execution in the local environment reproduced the numerical summaries to floating-point precision.

## Estimator

Training matrices B and X contain baseline and proposed-extension payoffs. Solve

`min_(a,b) ||1 - B a - X b||^2 / T + lambda ||b||^2`.

Let `P_B = B (B'B)^(-1) B'`, `a0 = (B'B)^(-1) B'1`, and `H=(I-P_B)X`. Then

`b = (H'H/T + lambda I)^(-1) H'1/T`,

`a = a0 - (B'B)^(-1) B'X b`.

This is partial ridge / profiled regression, not a claim that profiling itself is new. Its use makes the economic comparison exact: at zero incremental effective dimension, the expanded estimator is exactly the same estimated baseline. At positive dimension, the new positions also induce a hedge adjustment to the baseline.

Complexity means only the smoother's managed-payoff effective dimension:

`C = rank(B) + tr[ Sigma_perp (Sigma_perp + lambda I)^(-1) ]`,

where `Sigma_perp=H'H/T`. This is a partially penalized, finite-rank variant. It does not automatically inherit every theorem for the manuscript's isotropically penalized infinite-dimensional estimator.

The uncentered projection H belongs to the response-one estimator. Do not confuse it with the centered covariance projection used in the population spanning identity `Delta SR^2 = alpha' Var(epsilon)^(-1) alpha`.

## Fixed real-data design

Official JKP USA capped-value-weighted characteristic-managed portfolio returns. Retain 57 complete portfolios whose cited paper appeared by 2004. Original baseline: `market_equity`, `be_me`, `ret_12_1`. These are three long-short tilts, not the Fama-French three-factor model. The full master return vector is fixed; excluded policy coordinates are set to zero. There are no hand-designed state predictors.

Normalize payoff units using RMS values fixed through December 1994. Train on the preceding 60, 120, or 240 completed months. For each outer year 2005-2025, use the first 60 months of the preceding decade to choose the added effective dimension by response-one loss, reserve the next 60 for confirmation, and evaluate the following 12 months. Refit coefficients monthly using past returns only. Disjoint chronological blocks are not necessarily statistically independent.

The full 2005-2025 period has already been examined in the project. These are exploratory chronological OOS results, not a pristine confirmatory holdout. A publication cutoff is not a reconstruction of historical data vintages. These are **characteristic-managed returns**, not raw stock-level characteristics or a CTF submission.

### Main result with the original baseline

At 120 months, baseline SR is -0.004, baseline plus the remaining value directions is 0.724, and baseline plus all extensions is 0.691. The weakness of the baseline is a limitation, not something to conceal. Group inference, including simultaneous bands, is in `results/paired_inference.csv`.

### Secondary market-inclusive baseline

Add the official JKP market return, hold the other choices fixed, and protect four baseline directions. This test was specified only after observing the weak original baseline.

| Training months | Market-inclusive baseline | + Value | + Investment/accruals | + All extensions |
|---|---:|---:|---:|---:|
| 120 | 0.442 | 0.914 | 0.688 | 0.981 |
| 240 | 0.333 | 1.033 | 0.603 | 1.068 |

At 120 months the all-extension Sharpe increment is 0.540; its simultaneous 95% interval within this four-extension diagnostic is [0.053, 1.027] for 12-month bootstrap blocks. The sign also survives the specified 6/24-month block checks. This band does not adjust for the entire research history. Source URL and SHA256 are recorded in `results/market_source.json`.

Additional value information contributes beyond book-to-market already present in the baseline. Further momentum is much less useful in this comparison. This does **not** identify a universally best feature list or establish that every individual value characteristic is necessary.

## Population experiment: same spectrum, same frontier

Two 27-dimensional Gaussian return economies share the **exact same second-moment matrix** D. Their means satisfy the same baseline and full response-one oracle losses, with annualized oracle SR 0.7 and 1.0 respectively. Only the location of the added mean signal changes: first four versus last four candidate eigendirections. Covariance is `D-mm'` and is positive definite in both cases.

Each replication estimates weights from T observations, selects incremental C using 120 separate validation observations, and evaluates the selected portfolio from exact population moments.

| Training months | Learned population SR: leading signal | Learned population SR: trailing signal |
|---|---:|---:|
| 60 | 0.564 | 0.442 |
| 120 | 0.690 | 0.534 |
| 240 | 0.780 | 0.609 |
| 480 | 0.857 | 0.699 |
| 960 | 0.901 | 0.795 |

At T=240 and total C=7, the paired improvement in response-one loss, divided by the oracle incremental opportunity value, is +0.623 for the leading signal and -0.171 for the trailing signal. This normalization compares to the estimated baseline from the same replication; it need not lie in [0,1].

This is a finite-rank alignment mechanism experiment, **not** identification of the source exponent r or confirmation of a nonparametric learning rate. In finite rank all targets can satisfy many source exponents with different radii. The Gaussian simulation also does not satisfy the bounded-support version of the manuscript's assumptions. No algorithm-independent impossibility claim follows from these simulations.

## Portfolio-manager algorithm and its failure mode

The main usable research primitive is `fit_extension`: preserve a baseline, estimate residual opportunity directions, apply spectral shrinkage only to the extension, and adjust the baseline hedge. Choose the effective-dimension budget strictly before the evaluation date.

The optional experimental admission gate requires a positive simultaneous lower bound on held-out response-one improvement. It admits extensions in only 5 of 21 years per history. At 120 months it attains SR 0.208 versus -0.004 for the baseline and 0.691 for all extensions. It is too restrictive in this experiment; it is **not** a proven superior allocation algorithm. The evidence favors testing continuous shrinkage rather than equating uncertainty with hard exclusion.

`recommend.py` exposes this gate as an auditable research API. The supplied as-of example concerns January 2026; it is not current. Its native response-one exposures are not stock orders. Some unscaled research payoffs fall below -1, so no limited-liability wealth curve or deployment claim is made.

## Reproduce

```bash
pip install -r incremental_opportunities/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s incremental_opportunities/tests -v
python incremental_opportunities/run.py --raw /path/to/archived/jkp_public_raw --out results --reps 400
python incremental_opportunities/market_check.py --out results
python incremental_opportunities/recommend.py --inputs results/inputs --as-of 2025-12-31 --out research_recommendation.json
```

The workflow reuses the archived JKP snapshot from the prior download artifact, not the previous study's results. The companion download bundle contains the source data needed to reproduce the run. Large replication-level tables and all monthly return series are in the `Incremental-Opportunity-Replication` workflow artifact. Small evidence tables are committed in `results/`.

For an already selected budget, the estimation primitive is simply:

```python
fit = fit_extension(training_returns, n_base=4, budgets=np.array([selected_extra_C]))
weights = fit.weights[:, 0]
# training_returns must contain only completed observations, in fixed units.
# Columns 0:4 are the baseline; the rest are the proposed extension.
```

## Sources

Jensen, Kelly, Pedersen (2023), *Is There a Replication Crisis in Finance?*, Journal of Finance 78, 2465-2518; official source https://jkpfactors.com/data. Public data license: **CC BY-NC 4.0, non-commercial research**.

Kozak, Nagel, Santosh (2020), *Shrinking the Cross Section*: residual economic contribution and shrinkage versus characteristic sparsity. Kelly, Malamud, Zhou (2024), *The Virtue of Complexity in Return Prediction*, and Kelly, Malamud (2025), *Understanding the Virtue of Complexity*: nested comparisons, representation and alignment. The comparison here is not a refutation of their estimator sequences.

The project's learnability theory motivates the economic questions; the profiled estimator and admission rule are research implementations to test, not additional theorems already proved by that theory.
