import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


def make_series(n=500, value=5.0, noise=0.1):
    """Helper: make a time-indexed series of n values around `value`."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(value, noise, n), index=idx)


# ─── Macro Stress ─────────────────────────────────────────────────────────────

def test_get_yield_curve_returns_dict_keys():
    from modules.market_pulse.indicators import get_yield_curve
    dgs2  = make_series(500, value=4.0, noise=0)
    dgs10 = make_series(500, value=4.5, noise=0)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[dgs2, dgs10]):
        result = get_yield_curve()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_yield_curve_spread_is_10y_minus_2y():
    from modules.market_pulse.indicators import get_yield_curve
    dgs2  = make_series(100, value=4.0, noise=0)
    dgs10 = make_series(100, value=4.5, noise=0)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[dgs2, dgs10]):
        result = get_yield_curve()
    assert abs(result["current"] - 0.5) < 0.01


def test_get_yield_curve_invert_is_true():
    from modules.market_pulse.indicators import get_yield_curve
    s = make_series(100)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[s, s]):
        result = get_yield_curve()
    assert result["invert"] is True


def test_get_fed_funds_returns_dict_keys():
    from modules.market_pulse.indicators import get_fed_funds
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=make_series(100)):
        result = get_fed_funds()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_fed_funds_current_is_last_value():
    from modules.market_pulse.indicators import get_fed_funds
    s = make_series(50, value=5.25, noise=0)
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=s):
        result = get_fed_funds()
    assert abs(result["current"] - 5.25) < 0.01


# ─── Fear & Sentiment ─────────────────────────────────────────────────────────

def test_get_vix_returns_dict_keys():
    from modules.market_pulse.indicators import get_vix
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({"^VIX": np.full(500, 18.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_vix_current_is_last_value():
    from modules.market_pulse.indicators import get_vix
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_df = pd.DataFrame({"^VIX": np.linspace(15.0, 25.0, 100)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix()
    assert abs(result["current"] - 25.0) < 0.01


def test_get_aaii_sentiment_returns_dict_keys():
    from modules.market_pulse.indicators import get_aaii_sentiment
    weekly_idx = pd.date_range("2023-01-01", periods=100, freq="W")
    bull = pd.Series(np.full(100, 40.0), index=weekly_idx)
    bear = pd.Series(np.full(100, 30.0), index=weekly_idx)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[bull, bear]):
        result = get_aaii_sentiment()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_aaii_sentiment_spread_is_bull_minus_bear():
    from modules.market_pulse.indicators import get_aaii_sentiment
    weekly_idx = pd.date_range("2023-01-01", periods=50, freq="W")
    bull = pd.Series(np.full(50, 40.0), index=weekly_idx)
    bear = pd.Series(np.full(50, 25.0), index=weekly_idx)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[bull, bear]):
        result = get_aaii_sentiment()
    assert abs(result["current"] - 15.0) < 0.01


def test_get_consumer_confidence_returns_dict_keys():
    from modules.market_pulse.indicators import get_consumer_confidence
    monthly = pd.Series(
        np.linspace(60, 80, 36),
        index=pd.date_range("2023-01-01", periods=36, freq="ME")
    )
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=monthly):
        result = get_consumer_confidence()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_consumer_confidence_invert_is_true():
    from modules.market_pulse.indicators import get_consumer_confidence
    s = pd.Series(np.linspace(60, 80, 36), index=pd.date_range("2023-01-01", periods=36, freq="ME"))
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=s):
        result = get_consumer_confidence()
    assert result["invert"] is True


# ─── Risk Appetite ────────────────────────────────────────────────────────────

def test_get_hy_credit_spread_returns_dict_keys():
    from modules.market_pulse.indicators import get_hy_credit_spread
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=make_series(500, value=4.0)):
        result = get_hy_credit_spread()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_gold_spy_ratio_returns_dict_keys():
    from modules.market_pulse.indicators import get_gold_spy_ratio
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({"GLD": np.full(500, 180.0), "SPY": np.full(500, 450.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_gold_spy_ratio()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_gold_spy_ratio_value_is_gld_over_spy():
    from modules.market_pulse.indicators import get_gold_spy_ratio
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_df = pd.DataFrame({"GLD": np.full(100, 180.0), "SPY": np.full(100, 450.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_gold_spy_ratio()
    assert abs(result["current"] - (180.0 / 450.0)) < 0.001


def test_get_dxy_returns_dict_keys():
    from modules.market_pulse.indicators import get_dxy
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({"DX-Y.NYB": np.full(500, 104.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_dxy()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


# ─── Global Heatmap ───────────────────────────────────────────────────────────

def test_get_global_heatmap_returns_dataframe():
    from modules.market_pulse.indicators import get_global_heatmap, HEATMAP_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({t: np.linspace(100, 110, 30) for t in HEATMAP_TICKERS}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_global_heatmap()
    assert isinstance(result, pd.DataFrame)


def test_get_global_heatmap_has_required_columns():
    from modules.market_pulse.indicators import get_global_heatmap, HEATMAP_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({t: np.linspace(100, 110, 30) for t in HEATMAP_TICKERS}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_global_heatmap()
    assert all(c in result.columns for c in ["ticker", "label", "1d", "1w", "1m"])


def test_get_global_heatmap_1d_return_correct():
    from modules.market_pulse.indicators import get_global_heatmap
    prices = np.linspace(100.0, 110.0, 30)
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({"SPY": prices}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_global_heatmap()
    spy_row = result[result["ticker"] == "SPY"].iloc[0]
    expected_1d = prices[-1] / prices[-2] - 1
    assert abs(spy_row["1d"] - expected_1d) < 1e-6
