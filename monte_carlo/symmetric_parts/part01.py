@dataclass(frozen=True)
class SobolevDimensionDesign:
    d: int
    modes: tuple[np.ndarray, ...]
    tau: np.ndarray
    l2_coef: np.ndarray
    z_eval: np.ndarray
    lambda_eval: np.ndarray
    theory_rate: float
    ridge_exponent: float


def make_sobolev_dimension_design(d: int) -> SobolevDimensionDesign:
    modes_list = mc.fourier_modes(d, mc.TARGET_PAIRS)
    tau = mc.fourier_eigenvalues(modes_list, mc.S)
    source = mc.nonlinear_source_coefficients(tau.size, mc.ECON.K, seed=20260818)
    base_coef = (tau ** (mc.R_SOURCE / 2.0))[:, None] * source
    base_m = sum(float(c @ mc.ECON.H @ c) for c in base_coef)
    scale = math.sqrt(mc.TARGET_M / base_m)
    coef = scale * base_coef
    z_eval = evaluation_grid(d, 13)
    lambda_eval = mc.fourier_l2_features(z_eval, modes_list) @ coef
    rate = 2.0 * mc.S * mc.R_SOURCE / (2.0 * mc.S * mc.R_SOURCE + d)
    ridge_exp = 2.0 * mc.S / (2.0 * mc.S * mc.R_SOURCE + d)
    return SobolevDimensionDesign(
        d=d,
        modes=tuple(modes_list),
        tau=tau,
        l2_coef=coef,
        z_eval=z_eval,
        lambda_eval=lambda_eval,
        theory_rate=rate,
        ridge_exponent=ridge_exp,
    )


SOBOLEV_DESIGNS = {d: make_sobolev_dimension_design(d) for d in D_LEVELS}
LINEAR_EVAL_GRIDS = {d: evaluation_grid(d, 13) for d in D_LEVELS}
LINEAR_EVAL_FEATURES = {d: mc.legendre_features(LINEAR_EVAL_GRIDS[d], 1) for d in D_LEVELS}
LINEAR_EVAL_LAMBDAS = {d: lambda_linear_dimension(LINEAR_EVAL_GRIDS[d], d, BASE_SR) for d in D_LEVELS}

