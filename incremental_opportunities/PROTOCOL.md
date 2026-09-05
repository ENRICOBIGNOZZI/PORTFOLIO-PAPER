# Incremental investment opportunities: research protocol, 2026-09-05

This replaces model horse races with two questions: what does an extension add beyond an existing portfolio, and how much of that value can be learned?

## Scope
Public official JKP USA capped-value-weighted characteristic-managed returns from the archived September 2026 snapshot. The investible return vector is fixed to the complete factors whose cited publication year is <=2004. Baseline policy: size (`market_equity`), book-to-market (`be_me`), momentum (`ret_12_1`). The same assets remain present in the master return vector; nested policy classes set excluded coefficients to zero. Add each bibliographic group, or all other directions, without changing states, dates or units. This is a finite-rank/static policy experiment, NOT raw stock-feature selection or a test of the nonlinear Matérn learning exponent. There are no manufactured state variables.

Fixed return-unit calibration uses RMS returns through December 1994. End date December 2025. Current-vintage history is not a point-in-time reconstruction. The 2005-2025 period was already examined in previous work: all new evidence is exploratory chronological out-of-sample evidence, not a pristine confirmatory holdout.

## Estimator and nesting
Profile the baseline out of the response-one loss, then apply ridge only to the added residual directions. Baseline parameters are unpenalized. For training matrices B and X, let P_B=B(B'B)^(-1)B'. Residual H=(I-P_B)X, and b_lambda=(H'H/T+lambda I)^(-1)H'1/T. The baseline correction is a=a0-(B'B)^(-1)B'X b_lambda. C=rank(B)+tr[(H'H/T)(H'H/T+lambda I)^(-1)]. C_extra=0 reproduces exactly the same estimated baseline. Full baseline-hedged and deliberately unhedged extensions are separately assessed; the latter is a diagnostic, not another tuning choice. No state-dependent normalization is used in the primary returns. Sharpe comparisons refer to native response-one policies. Constant-risk frontier translations are descriptive, not known oracle gaps.

## Empirical design
Training histories 60,120,240 months. Incremental complexity grid: 0,0.5,1,2,4,8,16,32, clipped below the residual rank. For each outer evaluation year 2005-2025: first preceding 60-month block selects C by mean response-one loss; following 60-month block is reserved for independent admission of the locked extension. Each monthly fit uses only completed preceding returns.

Report (i) baseline; (ii) each fixed group and all-directions extension, C chosen only in the first block; (iii) residual-opportunity gate: familywise 95% simultaneous circular-block-bootstrap lower bound of mean loss improvement in the second block must exceed zero. If none passes, return baseline. Select the qualifying extension with largest lower bound; lock group and C for the next year. The gate is experimental and is not implied by the minimax theorem. Bootstrap gate: 400 replications, block length 12; final paired evidence: 2000 replications, blocks 6/12/24.

The main window is 120 months. Other windows are comparisons, never selected ex post. Individual extensions at fixed C_extra=0.5 may be described, but do not feed a test-selected feature ranking into the algorithm.

## Controlled population experiment
Fixed 27-dimensional second-moment spectrum, 3 baseline coordinates and 24 candidate coordinates. Same population maximum Sharpe and same baseline frontier in both economies. Rotate the incremental optimal payoff signal between leading and trailing candidate eigen-directions, holding second moments fixed. Draw Gaussian returns with covariance D-mm'; report this as a finite-rank Gaussian mechanism experiment, not verification of the bounded-return/nonparametric theorem. Fit histories 60,120,240,480,960 and validate on a separate 120 observations. Evaluate population regret and Sharpe from exact known moments. 400 paired replications per history. Do not clip negative or >100% realized recovery fractions.

## Outputs and interpretation
Matched dates/units, profile-ridge algebra, baseline nesting, no-future-input tests, actual returns and admission logs, exact simulation oracles, CSV and concise financial report. Distinguish population opportunity value, estimated incremental gain, and admission decision. Do not equate empirical effective dimension with identified alpha count, infer source exponent r, or claim a universal interior Sharpe optimum.
