# Learning Portfolio Decisions

Reproducible code for the direct portfolio learning paper.

## Repository layout

- `empirical_jkp_ctf/`: Jensen–Kelly–Pedersen Common Task Framework implementation of the paper's complexity tests. It contains the single-Matérn headline design, the four requested empirical figures plus kernel robustness, a WRDS downloader for the official CTF tables, a CTF-compatible portfolio-kernel submission file, CSV audit outputs, and smoke tests.
- `monte_carlo/`: readable symmetric Monte Carlo code for the theoretical learning rates, including the Linear vs Sobolev SNR / asset-dimension / state-dimension experiments and bootstrap post-processing.
- `bundles/PORTFOLIO_PAPER_EMPIRICAL_SPECTRAL_CODE.zip`: complete frozen source tree for the earlier real-data Direct Gaussian/Matérn KRR application and the empirical spectral-decay analysis.
- `restore_empirical_spectral_source.py`: verifies the frozen bundle and restores the full `empirical/` and `spectral_decay/` source directories after cloning.

The JKP/CTF stock-level data are licensed and are therefore not committed. See `empirical_jkp_ctf/README.md` for the official WRDS route and the exact commands that generate the paper figures.

Restore the frozen earlier empirical/spectral sources with:

```bash
python restore_empirical_spectral_source.py
```

The bundle contains the full empirical project (`config.py`, `src/`, `scripts/`, `tests/`, LaTeX helpers and audit metadata) plus the spectral code fitting `A j^{-b}`, `A exp(-cj)` and `A j^{-b} exp(-cj)`, including block-bootstrap inference and histogram/fit figures.

Licensed WRDS/JKP/CRSP data, credentials, and large proprietary caches are intentionally excluded. Reproducing the real-data outputs requires an independently authorized local data cache.

Frozen empirical/spectral bundle SHA256:

`1a8982011fe2fc8bbe554e2fbdae90717a3bd90124356c479769d1b1f1edec99`
