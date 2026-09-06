# Fixed-Matérn learnability experiment

This branch contains the real-data design used for the revised portfolio-learnability paper. The empirical section intentionally holds the representation fixed and varies only regularization/effective complexity after a pre-test length-scale choice.

## Data and chronology

- Official public U.S. Jensen–Kelly–Pedersen characteristic-managed long-short portfolios.
- 153 capped-value-weighted characteristic portfolios.
- State: lagged 12-month means and volatilities of seven original-literature groups (14 coordinates).
- Calibration through 1994-12.
- Validation: 1995-01 through 2004-12.
- Test: 2005-01 through 2025-12 (252 months).
- Main rolling training history: 120 months.

## Representation and tuning

One Matérn-3/2 representation is used in the empirical section. The length scale is selected only on the validation response-one loss from 19 multiples of the pre-test median state distance:

`0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4, 5.6, 8, 11.2, 16, 22.6, 32, 45, 64`.

The effective-complexity grid is

`0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 118`.

The selected length scale is 16 times the pre-test median distance, and validation selects effective complexity C=28.

## Headline results

At the validation-selected model:

- validation response-one loss: 0.529571
- test response-one loss: 0.899043
- test annualized Sharpe: 1.227757
- mean effective complexity: 28
- mean ridge level: 0.00033352

The ex-post test-loss minimum is at C=20 with loss 0.886828 and Sharpe 1.226206. Thus the validation-selected C=28 lies close to the subsequent test-loss optimum relative to the admissible range 0.25–118.

Near interpolation (C=118), median in-sample Sharpe is about 172.45, while test Sharpe is 0.3654 and test response-one loss rises to 3.1528. This is the main empirical fit-versus-learnability result.

## Training-history comparative statics

Holding the selected Matérn geometry fixed, validation selects:

| T months | selected C | validation loss | test loss | test Sharpe |
|---:|---:|---:|---:|---:|
| 60 | 20 | 0.597666 | 0.991383 | 0.933951 |
| 120 | 28 | 0.529571 | 0.899043 | 1.227757 |
| 180 | 32 | 0.533039 | 0.902046 | 1.249002 |
| 240 | 40 | 0.501703 | 0.902183 | 1.224059 |

Selected effective complexity increases with available market history, while realized test Sharpe need not be monotone.

## Spectrum

For the selected Matérn representation, the average training-trace shares are highly concentrated: rank 1 accounts for 48.0%, ranks 1–2 for 68.4%, ranks 1–5 for 82.2%, ranks 1–10 for 90.1%, and ranks 1–20 for 95.3%.

## Interpretation

This is not a real-data kernel horse race and is not used to estimate population representation error. Those comparisons are moved to controlled simulations where the data-generating process and oracle policy are known. The real-data exercise asks only how much effective portfolio complexity a finite history supports inside one rich fixed representation.
