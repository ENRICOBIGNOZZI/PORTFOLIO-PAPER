def linear_snr_task(T: int, rep: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(common_seed(seed, 101, T, rep))
    z = cs.simulate_state_fast(T, 3, BASE_RHO, rng)
    factor_shocks = rng.standard_normal((T, mc.ECON.K))
    idio_shocks = rng.standard_normal((T, mc.ECON.N))
    z_eval = LINEAR_EVAL_GRIDS[3]
    features = mc.legendre_features(z, 1)
    eval_features = LINEAR_EVAL_FEATURES[3]
    rows = []
    for sr in SNR_LEVELS:
        lam = lambda_linear_dimension(z, 3, sr)
        lam_eval = lambda_linear_dimension(z_eval, 3, sr)
        returns = sample_returns_common(lam, factor_shocks, idio_shocks, mc.ECON)
        ridge = RIDGE_LINEAR_SCALE / T
        coef = mc.fit_direct_policy(features, returns, ridge)
        iters, residual = 0, 0.0
        weights = mc.evaluate_policy(eval_features, coef)
        metrics = population_metrics(weights, lam_eval, mc.ECON, sr)
        rows.append({
            "experiment": "signal_to_noise",
            "model": "Linear",
            "oracle_sr_annual": sr,
            "portfolio_snr": portfolio_snr(sr),
            "T": T,
            "rep": rep,
            "N": mc.ECON.N,
            "d": 3,
            "theory_rate": 1.0,
            "ridge": ridge,
            "ridge_scale": RIDGE_LINEAR_SCALE,
            "n_features": features.shape[1],
            "n_coefficients": mc.ECON.N * features.shape[1],
            "solver_iterations": iters,
            "solver_residual": residual,
            **metrics,
        })
    return rows

