# Learning Portfolio Decisions — symmetric Monte Carlo

This directory contains the paper-facing Monte Carlo for direct portfolio learning.
The final design is deliberately symmetric:

| Population policy | Signal-to-noise | Assets `N` | State dimension `d` |
|---|---|---|---|
| Correctly specified linear | fixed-dimensional rate `T^-1` | fixed-dimensional rate `T^-1` | fixed-dimensional rate `T^-1` |
| Correctly specified Sobolev | structural rate `T^-2/3` at `d=3` | structural rate `T^-2/3` for each fixed `N` | rate `T^{-6/(6+d)}` |

The common DGP is a persistent conditional factor economy with one observed state `Z_t`, conditional factor premia `lambda(Z_t)`, constant conditional return second moment, and an analytically known population tangency policy. Population performance is evaluated by deterministic 8,192-point Sobol integration.

Key state-dimension estimates: `d=1`: 0.879 vs theory 0.857; `d=2`: 0.760 vs 0.750; `d=3`: 0.671 vs 0.667.

Run `python run_symmetric_paper_mc.py --jobs 3` for the symmetric rate layer or `python run_final_paper_mc.py --jobs 3` for the full Monte Carlo pipeline.

## Simulated complexity figures

`simulated_complexity_figures.py` generates the paper-facing illustrations for the finite-sample complexity mechanism: regret decomposition, oracle versus learned Sharpe, the complexity scaling law, in-sample versus out-of-sample Sharpe, and nominal versus effective complexity.

Run `python simulated_complexity_figures.py --quick` for a smoke test or `python simulated_complexity_figures.py --reps 2500` for the full version. Publication-ready PDF/PNG/SVG outputs are written to `figures_simulated/`; the same directory contains committed SVG previews and a figure-by-figure README.
