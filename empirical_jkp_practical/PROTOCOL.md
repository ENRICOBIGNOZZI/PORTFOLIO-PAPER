# Practical portfolio learnability - locked exploratory protocol

Protocol fixed before the new performance/selection runs, 2026-09-05. This is a new analysis, not a replication certification of numerical claims in earlier PDFs.

## Data and scope
Official Jensen-Kelly-Pedersen monthly USA characteristic-managed long-short portfolio returns, capped-value-weighted, current vintage through December 2025. Supplied returns already have their literature orientation: do NOT multiply by direction again. This is allocation across characteristic portfolios, NOT a stock-level CTF submission. No stock panel or underlying stock trades are available in this session. Save URL, retrieval date, SHA256 and data license (CC BY-NC 4.0; research only).

## Chronology
Data begin January 1973 for this experiment. Select universe using completeness and nonzero variance through December 1994 ONLY. State variables are lagged twelve-month means and volatilities of seven broad original-literature groups from the official factor-details file (not the 13 estimated JKP clusters). Freeze state calibration and median kernel length scale using information through December 1994. Validation: January 1995-December 2004. Test: January 2005-December 2025. Training histories: 60,120,240 months on common evaluation dates. No contemporaneous payoff enters current weights.

The full current feature library is retrospective. Separately evaluate a fixed library with publication year <=2004. This is a publication-date sensitivity, not a historical-data-vintage reconstruction.

## Estimation and terminology
Response-one ridge uses G_ts=K(Z_t,Z_s) R_t'R_s / T. Complexity C=trace[G(G+lambda I)^-1], never feature count. A predeclared C grid determines lambda by bisection on training eigenvalues, with rank caps reported. This empirical, data-dependent tuning is NOT a structural estimator of b or r. Canonical kernel: exact Matern 3/2. Constant, linear, RBF and paired nested RFF Matern approximations are representation sensitivities.

Select C by validation Sharpe after causal risk scaling; also report native-loss selection. Test maxima are descriptive, never selections. Risk target: 10% annualized; trailing covariance with 10% diagonal shrinkage; absolute factor-sleeve exposure sum <=3. Report realized risk and native loss separately. Scaling changes the deployed policy; C labels the trained core.

## Feature selection
- Drop each investment-sleeve group, keep state fixed: retuned C and matched-C comparisons.
- Drop state-predictor groups, keep investment universe fixed.
- Leave-one-characteristic-out on validation; rank by full-minus-retuned-drop validation Sharpe. Freeze ranking and evaluate top-8,16,32,64 libraries and full library on test. Feature-count selection also uses validation only.
- Prespecified finance library and pre-2005-publication library.
- Compare stand-alone factor Sharpe ranks with joint portfolio contribution ranks. These are correlated-feature ablations, not causal effects or unique alpha attribution.

## Diagnostics
IS/OOS C curves, validation choice, C versus training length, kernel comparison, nominal size versus C, paired block-bootstrap uncertainty, and test-implied capital-allocation lines. Lines are descriptive risk-return transformations, not measured distances from a known population frontier. Bootstrap uses 12-month circular blocks with 6/24 sensitivity and simultaneous family bands where relevant. OOS eigen-rank-band contributions are diagnostics, not source-exponent estimates.

## Trading-cost stress
Charges: 0,10,25,50 bps times sum|sleeve_weight_t-sleeve_weight_(t-1)|, with C retuned on validation. These omit internal factor turnover, stock netting, spreads, borrowing, impact and capacity. They are NOT net stock-level returns. No universal ordering of net-optimal and gross-optimal C is imposed.

## Delivery
Actual results/weights/CSV rankings, reproducible code, figures and report; causal tests; PR without changing main. Stock-level conclusions require authorized CTF data and independent rerunning of the same selection protocol.
