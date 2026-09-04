from __future__ import annotations

import numpy as np
import pandas as pd

from core import KernelSpec, build_managed_feature_cache, rolling_response_one_curve


def mock_chars(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1980-01-31", periods=180, freq="ME")
    rows = []
    n_assets = 40
    for t, d in enumerate(dates):
        x = rng.normal(size=(n_assets, 4))
        signal = 0.01 * (np.sin(x[:, 0]) + 0.5 * x[:, 1] * x[:, 2])
        r = signal + rng.normal(scale=0.06, size=n_assets)
        for i in range(n_assets):
            rows.append(
                {
                    "id": i,
                    "eom": d,
                    "ret_exc_lead1m": r[i],
                    "ctff_test": t >= 120,
                    "x1": x[i, 0],
                    "x2": x[i, 1],
                    "x3": x[i, 2],
                    "x4": x[i, 3],
                }
            )
    chars = pd.DataFrame(rows)
    features = pd.DataFrame({"features": ["x1", "x2", "x3", "x4"]})
    return chars, features


def test_pipeline() -> None:
    chars, features = mock_chars()
    names = features["features"].tolist()
    spec = KernelSpec(name="matern", nu=1.5, max_features=64, seed=3)
    cache = build_managed_feature_cache(chars, names, spec, min_assets=10)
    F = cache.matrix(64)
    curve = rolling_response_one_curve(F, cache.dates, cache.test_mask, window=60)
    assert len(curve.lambdas) == 61
    assert np.isfinite(curve.avg_complexity).all()
    assert np.isfinite(curve.oos_sharpe).any()
    assert np.all(curve.avg_complexity >= -1e-10)
    assert np.all(curve.avg_complexity <= 60 + 1e-8)


if __name__ == "__main__":
    test_pipeline()
    print("smoke test passed")
