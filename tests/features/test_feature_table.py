import numpy as np
import pandas as pd
from unittest.mock import patch


REQUIRED_COLUMNS = [
    "vol_regime_score", "vol_regime_label",
    "momentum_short", "momentum_long",
    "mean_reversion", "trend_strength",
    "macro_stress",
    "volume_regime_score", "volume_regime_label",
    "fiftytwo_week_pos",
    "relative_strength",
]


def _make_prices(n=300, seed=42):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(seed)
    returns = np.random.normal(0, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + returns), index=idx)


def _patch_fg(prices):
    return patch(
        "modules.integration.regimes.get_fear_greed_history",
        return_value=pd.Series(50.0, index=prices.index),
    )


def test_feature_table_has_required_columns():
    from modules.features.feature_table import build
    prices = _make_prices()
    with _patch_fg(prices):
        result = build(prices)
    assert all(c in result.columns for c in REQUIRED_COLUMNS)


def test_feature_table_index_matches_prices():
    from modules.features.feature_table import build
    prices = _make_prices()
    with _patch_fg(prices):
        result = build(prices)
    assert result.index.equals(prices.index)


def test_feature_table_handles_missing_benchmark():
    from modules.features.feature_table import build
    prices = _make_prices()
    with _patch_fg(prices):
        result = build(prices, benchmark=None)
    assert result["relative_strength"].isna().all()
