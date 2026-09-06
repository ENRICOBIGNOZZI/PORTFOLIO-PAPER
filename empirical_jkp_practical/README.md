# Fixed-Matérn paper experiment

This branch adds the fixed-representation empirical experiment used in the revised portfolio-learnability paper.

`fixed_matern_deep.py` is designed to run with the public-JKP data preparation and portfolio engine developed on branch `research/jkp-practical-learnability`. That branch contains `engine.py` and `fetch_public.py`, which download only the public JKP characteristic-managed portfolio data and construct the locked chronological state.

The revised paper intentionally does **not** use a real-data kernel horse race and does **not** use forecast-then-optimize benchmarks. Representation comparisons and the decomposition of representation versus learning error are carried out only in controlled simulations where the oracle policy is known.

See `results_fixed_matern_20260906/RESULTS.md` for the locked chronology, hyperparameter grids, and headline numbers.