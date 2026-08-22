def state_dimension_task(T: int, rep: int, seed: int,
                         include_sobolev_d3: bool = False) -> list[dict]:
    rng = np.random.default_rng(common_seed(seed, 121, T, rep))
    z3 = cs.simulate_state_fast(T, 3, BASE_RHO, rng)
    factor_shocks = rng.standard_normal((T, mc.ECON.K))
    idio_shocks = rng.standard_normal((T, mc.ECON.N))
    rows: list[dict] = []

    for d in D_LEVELS:
        z = z3[:, :d]

        # Correctly specified linear policy.
        lam_l = lambda_linear_dimension(z, d, BASE_SR)
        z_eval_l = LINEAR_EVAL_GRIDS[d]
        lam_eval_l = LINEAR_EVAL_LAMBDAS[d]
        returns_l = sample_returns_common(lam_l, factor_shocks, idio_shocks, mc.ECON)
        features_l = mc.legendre_features(z, 1)
        eval_l = LINEAR_EVAL_FEATURES[d]
        ridge_l = RIDGE_LINEAR_SCALE / T
        coef_l = mc.fit_direct_policy(features_l, returns_l, ridge_l)
        it_l, res_l = 0, 0.0
        weights_l = mc.evaluate_policy(eval_l, coef_l)
        metrics_l = population_metrics(weights_l, lam_eval_l, mc.ECON, BASE_SR)
        rows.append({
            "experiment": "state_dimension",
            "model": "Linear",
            "oracle_sr_annual": BASE_SR,
            "portfolio_snr": portfolio_snr(BASE_SR),
            "T": T,
            "rep": rep,
            "N": mc.ECON.N,
            "d": d,
            "theory_rate": 1.0,
            "ridge": ridge_l,
            "ridge_scale": RIDGE_LINEAR_SCALE,
            "n_features": features_l.shape[1],
            "n_coefficients": mc.ECON.N * features_l.shape[1],
            "solver_iterations": it_l,
            "solver_residual": res_l,
            **metrics_l,
        })

        # d=3 is already available from the high-replication validated baseline.
        if d == 3 and not include_sobolev_d3:
            continue
        design = SOBOLEV_DESIGNS[d]
        lam_s = lambda_sobolev_dimension(z, design)
        returns_s = sample_returns_common(lam_s, factor_shocks, idio_shocks, mc.ECON)
        modes = list(estimation_modes(T, d))
        features_s = mc.flexible_features(z, modes, mc.S)
        eval_s = sobolev_eval_features(T, d)
        ridge_s = RIDGE_SOBOLEV_SCALE * T ** (-design.ridge_exponent)
        coef_s = mc.fit_direct_policy(features_s, returns_s, ridge_s)
        it_s, res_s = 0, 0.0
        weights_s = mc.evaluate_policy(eval_s, coef_s)
        metrics_s = population_metrics(weights_s, design.lambda_eval, mc.ECON, BASE_SR)
        rows.append({
            "experiment": "state_dimension",
            "model": "Sobolev",
            "oracle_sr_annual": BASE_SR,
            "portfolio_snr": portfolio_snr(BASE_SR),
            "T": T,
            "rep": rep,
            "N": mc.ECON.N,
            "d": d,
            "theory_rate": design.theory_rate,
            "ridge": ridge_s,
            "ridge_scale": RIDGE_SOBOLEV_SCALE,
            "ridge_exponent": design.ridge_exponent,
            "n_features": features_s.shape[1],
            "n_coefficients": mc.ECON.N * features_s.shape[1],
            "solver_iterations": it_s,
            "solver_residual": res_s,
            **metrics_s,
        })
    return rows

