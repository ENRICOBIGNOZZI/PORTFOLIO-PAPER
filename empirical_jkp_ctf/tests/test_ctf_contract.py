from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ctf_submission_matern", HERE / "ctf_submission_matern.py"
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
main = MOD.main


def mock_ctf(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=40, freq="ME")
    n_assets = 24
    rows: list[dict[str, object]] = []
    for t, d in enumerate(dates):
        x = rng.normal(size=(n_assets, 4))
        signal = 0.008 * (np.sin(x[:, 0]) + 0.4 * x[:, 1] * x[:, 2])
        ret = signal + rng.normal(scale=0.05, size=n_assets)
        for i in range(n_assets):
            rows.append(
                {
                    "id": i,
                    "eom": d,
                    "ret_exc_lead1m": ret[i],
                    "ctff_test": t >= 30,
                    "x1": x[i, 0],
                    "x2": x[i, 1],
                    "x3": x[i, 2],
                    "x4": x[i, 3],
                }
            )
    chars = pd.DataFrame(rows)
    features = pd.DataFrame({"features": ["x1", "x2", "x3", "x4"]})
    daily = pd.DataFrame(columns=["id", "date", "eom", "ret_exc"])
    return chars, features, daily


def _sorted(x: pd.DataFrame) -> pd.DataFrame:
    return x.sort_values(["eom", "id"]).reset_index(drop=True)


def test_schema_determinism_and_test_dates() -> None:
    chars, features, daily = mock_ctf()
    out1 = _sorted(main(chars.copy(), features.copy(), daily.copy()))
    out2 = _sorted(main(chars.copy(), features.copy(), daily.copy()))

    assert list(out1.columns) == ["id", "eom", "w"]
    assert len(out1) > 0
    assert out1[["id", "eom"]].duplicated().sum() == 0
    assert np.isfinite(out1["w"].to_numpy()).all()
    pd.testing.assert_frame_equal(out1, out2, check_exact=False, rtol=1e-12, atol=1e-12)

    test_dates = set(pd.to_datetime(chars.loc[chars["ctff_test"], "eom"]).unique())
    assert set(pd.to_datetime(out1["eom"]).unique()) <= test_dates


def test_truncation_no_lookahead() -> None:
    chars, features, daily = mock_ctf()
    all_test_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(chars.loc[chars["ctff_test"], "eom"]).unique())
    )
    cut = all_test_dates[3]

    full = _sorted(main(chars.copy(), features.copy(), daily.copy()))
    full_at_cut = _sorted(full.loc[pd.to_datetime(full["eom"]) == cut])

    truncated_chars = chars.loc[pd.to_datetime(chars["eom"]) <= cut].copy()
    trunc = _sorted(main(truncated_chars, features.copy(), daily.copy()))
    trunc_at_cut = _sorted(trunc.loc[pd.to_datetime(trunc["eom"]) == cut])

    pd.testing.assert_frame_equal(
        full_at_cut, trunc_at_cut, check_exact=False, rtol=1e-10, atol=1e-12
    )


def test_current_and_future_returns_do_not_change_current_weights() -> None:
    chars, features, daily = mock_ctf()
    test_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(chars.loc[chars["ctff_test"], "eom"]).unique())
    )
    cut = test_dates[2]

    base = _sorted(main(chars.copy(), features.copy(), daily.copy()))
    base_cut = _sorted(base.loc[pd.to_datetime(base["eom"]) == cut])

    shocked = chars.copy()
    date = pd.to_datetime(shocked["eom"])
    # ret_exc_lead1m at cut is a future payoff relative to the cut-date decision.
    # Later returns are also unavailable at cut. Neither may change cut-date weights.
    shocked.loc[date >= cut, "ret_exc_lead1m"] = (
        10.0 + np.arange((date >= cut).sum(), dtype=float)
    )
    shocked_out = _sorted(main(shocked, features.copy(), daily.copy()))
    shocked_cut = _sorted(shocked_out.loc[pd.to_datetime(shocked_out["eom"]) == cut])

    pd.testing.assert_frame_equal(
        base_cut, shocked_cut, check_exact=False, rtol=1e-10, atol=1e-12
    )


def test_validator_fallback_without_active_test_flag() -> None:
    chars, features, daily = mock_ctf()
    chars["ctff_test"] = False
    out = _sorted(main(chars, features, daily))
    assert len(out) > 0
    assert pd.to_datetime(out["eom"]).nunique() == 1
    assert pd.Timestamp(pd.to_datetime(out["eom"]).iloc[0]) == pd.Timestamp(chars["eom"].max())
    assert np.isfinite(out["w"].to_numpy()).all()


if __name__ == "__main__":
    test_schema_determinism_and_test_dates()
    test_truncation_no_lookahead()
    test_current_and_future_returns_do_not_change_current_weights()
    test_validator_fallback_without_active_test_flag()
    print("CTF contract tests passed")
