def linear_n_task(T: int, rep: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(common_seed(seed, 111, T, rep))
    z = cs.simulate_state_fast(T, 3, BASE_RHO, rng)
    lam = lambda_linear_dimension(z, 3, BASE_SR)
    z_eval = LINEAR_EVAL_GRIDS[3]
    lam_eval = LINEAR_EVAL_LAMBDAS[3]
    features = mc.legendre_features(z, 1)
    eval_features = LINEAR_EVAL_FEATURES[3]
    factor_shocks = rng.standard_normal((T, mc.ECON.K))
    idio_shocks = rng.standard_normal((T, max(N_LEVELS)))
    rows = []
    for n in N_LEVELS:
        econ = cs.ECONOMIES[n]
        returns = sample_returns_common(lam, factor_shocks, idio_shocks, econ)
        ridge = RIDGE_LINEAR_SCALE / T
        coef = mc.fit_direct_policy(features, returns, ridge)
        iters, residual = 0, 0.0
        weights = mc.evaluate_policy(eval_features, coef)
        metrics = population_metrics(weights, lam_eval, econ, BASE_SR)
        rows.append({
            "experiment": "asset_dimension",
            "model": "Linear",
            "oracle_sr_annual": BASE_SR,
            "portfolio_snr": portfolio_snr(BASE_SR),
            "T": T,
            "rep": rep,
            "N": n,
            "d": 3,
            "theory_rate": 1.0,
            "ridge": ridge,
            "ridge_scale": RIDGE_LINEAR_SCALE,
            "n_features": features.shape[1],
            "n_coefficients": n * features.shape[1],
            "solver_iterations": iters,
            "solver_residual": residual,
            **metrics,
        })
    return rows

