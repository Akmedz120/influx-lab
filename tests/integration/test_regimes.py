import numpy as np
import pandas as pd
from unittest.mock import patch


# ─── classify_regime ──────────────────────────────────────────────────────────

def test_classify_regime_extreme_greed():
    from modules.integration.regimes import classify_regime
    assert classify_regime(0.0) == "Extreme Greed"
    assert classify_regime(10.0) == "Extreme Greed"
    assert classify_regime(19.9) == "Extreme Greed"


def test_classify_regime_greed():
    from modules.integration.regimes import classify_regime
    assert classify_regime(20.0) == "Greed"
    assert classify_regime(30.0) == "Greed"
    assert classify_regime(39.9) == "Greed"


def test_classify_regime_neutral():
    from modules.integration.regimes import classify_regime
    assert classify_regime(40.0) == "Neutral"
    assert classify_regime(50.0) == "Neutral"
    assert classify_regime(59.9) == "Neutral"


def test_classify_regime_fear():
    from modules.integration.regimes import classify_regime
    assert classify_regime(60.0) == "Fear"
    assert classify_regime(70.0) == "Fear"
    assert classify_regime(79.9) == "Fear"


def test_classify_regime_extreme_fear():
    from modules.integration.regimes import classify_regime
    assert classify_regime(80.0) == "Extreme Fear"
    assert classify_regime(100.0) == "Extreme Fear"


# ─── get_regime_history ────────────────────────────────────────────────────────

def test_get_regime_history_preserves_index():
    from modules.integration.regimes import get_regime_history
    idx = pd.date_range("2023-01-01", periods=5, freq="B")
    scores = pd.Series([10.0, 30.0, 50.0, 70.0, 90.0], index=idx)
    result = get_regime_history(scores)
    assert list(result.index) == list(idx)


def test_get_regime_history_correct_labels():
    from modules.integration.regimes import get_regime_history
    scores = pd.Series([10.0, 30.0, 50.0, 70.0, 90.0])
    result = get_regime_history(scores)
    assert list(result) == ["Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear"]


# ─── get_regime_asset_stats ────────────────────────────────────────────────────

def test_get_regime_asset_stats_returns_dataframe():
    from modules.integration.regimes import get_regime_asset_stats
    idx = pd.date_range("2023-01-01", periods=20, freq="B")
    regime_history = pd.Series(["Fear"] * 20, index=idx)
    prices = pd.DataFrame({"SPY": np.linspace(400, 420, 20)}, index=idx)
    result = get_regime_asset_stats(regime_history, prices)
    assert isinstance(result, pd.DataFrame)


def test_get_regime_asset_stats_has_all_five_regimes():
    from modules.integration.regimes import get_regime_asset_stats
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    labels = (["Extreme Greed"] * 20 + ["Greed"] * 20 +
              ["Neutral"] * 20 + ["Fear"] * 20 + ["Extreme Fear"] * 20)
    regime_history = pd.Series(labels, index=idx)
    prices = pd.DataFrame({"SPY": np.linspace(400, 500, 100)}, index=idx)
    result = get_regime_asset_stats(regime_history, prices)
    expected = {"Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear"}
    assert set(result.index) == expected


def test_get_regime_asset_stats_nan_for_sparse_regime():
    from modules.integration.regimes import get_regime_asset_stats
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    labels = ["Fear"] * 3 + ["Neutral"] * 7
    regime_history = pd.Series(labels, index=idx)
    prices = pd.DataFrame({"SPY": np.linspace(400, 410, 10)}, index=idx)
    result = get_regime_asset_stats(regime_history, prices)
    assert np.isnan(result.loc["Fear", "SPY"])


# ─── get_fear_greed_history ────────────────────────────────────────────────────

def _make_ind(idx, values, invert=False):
    return {
        "series": pd.Series(list(values), index=idx),
        "current": float(list(values)[-1]),
        "invert": invert,
        "label": "Test",
        "unit": "x",
    }


def test_get_fear_greed_history_returns_series():
    from modules.integration.regimes import get_fear_greed_history
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    ind = _make_ind(idx, range(100))
    targets = [
        "modules.market_pulse.indicators.get_vix",
        "modules.market_pulse.indicators.get_hy_credit_spread",
        "modules.market_pulse.indicators.get_dxy",
        "modules.market_pulse.indicators.get_yield_curve",
        "modules.market_pulse.indicators.get_gold_spy_ratio",
        "modules.market_pulse.indicators.get_aaii_sentiment",
    ]
    patches = [patch(t, return_value=ind) for t in targets]
    for p in patches:
        p.start()
    try:
        result = get_fear_greed_history()
    finally:
        for p in patches:
            p.stop()
    assert isinstance(result, pd.Series)
    assert len(result) > 0
    assert result.between(0, 100).all()


def test_get_fear_greed_history_excludes_sparse_dates():
    from modules.integration.regimes import get_fear_greed_history
    # Two indicators on different date ranges — overlap dates should be included, sparse excluded
    idx_a = pd.date_range("2023-01-01", periods=50, freq="B")
    idx_b = pd.date_range("2023-01-01", periods=10, freq="B")
    ind_a = _make_ind(idx_a, range(50))
    ind_b = _make_ind(idx_b, range(10))
    # 5 indicators return ind_a, 1 returns ind_b — all 50 dates have ≥3 indicators
    with patch("modules.market_pulse.indicators.get_vix",              return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_hy_credit_spread", return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_dxy",              return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_yield_curve",      return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_gold_spy_ratio",   return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_aaii_sentiment",   return_value=ind_b):
        result = get_fear_greed_history()
    # All 50 dates should be included (5 indicators cover all 50 dates)
    assert len(result) == 50
