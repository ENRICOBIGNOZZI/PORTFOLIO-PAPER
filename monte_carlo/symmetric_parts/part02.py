@lru_cache(maxsize=None)
def estimation_modes(T: int, d: int) -> tuple[np.ndarray, ...]:
    j_nonlin = sobolev_feature_count(T, d)
    return tuple(mc.fourier_modes(d, (j_nonlin - 1) // 2))


@lru_cache(maxsize=None)
def sobolev_eval_features(T: int, d: int) -> np.ndarray:
    return mc.flexible_features(SOBOLEV_DESIGNS[d].z_eval, list(estimation_modes(T, d)), mc.S)


def lambda_sobolev_dimension(z: np.ndarray, design: SobolevDimensionDesign) -> np.ndarray:
    return mc.fourier_l2_features(z, list(design.modes)) @ design.l2_coef


def sobolev_feature_count(T: int, d: int, ridge_scale: float = RIDGE_SOBOLEV_SCALE) -> int:
    ridge_exp = 2.0 * mc.S / (2.0 * mc.S * mc.R_SOURCE + d)
    ridge = ridge_scale * T ** (-ridge_exp)
    effective = ridge ** (-d / (2.0 * mc.S))
    J = int(math.ceil(2.5 * effective))
    J = max(17, min(71, J))
    if J % 2 == 0:
        J += 1
    return J


def population_metrics(weights: np.ndarray, lambda_eval: np.ndarray,
                       econ: mc.Economy, oracle_sr: float) -> dict[str, float]:
    return cs.population_metrics_econ(weights, lambda_eval, econ, oracle_sr)


def sample_returns_common(lambda_z: np.ndarray, factor_shocks: np.ndarray,
                          idio_shocks: np.ndarray, econ: mc.Economy) -> np.ndarray:
    return cs.sample_returns_with_common_shocks(lambda_z, factor_shocks, idio_shocks, econ)


def fit_policy(features: np.ndarray, returns: np.ndarray, ridge: float,
               econ: mc.Economy) -> tuple[np.ndarray, int, float]:
    return cs.fit_direct_policy_iterative(features, returns, ridge, econ)

