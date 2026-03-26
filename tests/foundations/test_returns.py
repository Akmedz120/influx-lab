import pytest
import pandas as pd
import numpy as np
from modules.foundations.returns import calculate_returns, annualize_return, annualize_volatility, compute_stats


@pytest.fixture
def prices():
    return pd.Series([100.0, 102.0, 101.0, 103.0, 105.0], name="SPY")


@pytest.fixture
def returns(prices):
    return calculate_returns(prices)


def test_returns_length(prices):
    r = calculate_returns(prices)
    assert len(r) == len(prices) - 1


def test_returns_first_value(prices):
    r = calculate_returns(prices)
    assert abs(r.iloc[0] - 0.02) < 1e-10


def test_returns_no_nans(prices):
    r = calculate_returns(prices)
    assert not r.isna().any()


def test_annualize_return_is_float(returns):
    result = annualize_return(returns)
    assert isinstance(result, float)


def test_annualize_volatility_non_negative(returns):
    result = annualize_volatility(returns)
    assert result >= 0


def test_annualize_volatility_weekly(returns):
    daily_vol = annualize_volatility(returns, frequency="daily")
    weekly_vol = annualize_volatility(returns, frequency="weekly")
    assert daily_vol != weekly_vol


def test_compute_stats_has_required_keys(returns):
    stats = compute_stats(returns)
    for key in ["mean_return", "volatility", "skewness", "kurtosis", "sharpe"]:
        assert key in stats


def test_compute_stats_sharpe_is_ratio(returns):
    stats = compute_stats(returns)
    expected = stats["mean_return"] / stats["volatility"] if stats["volatility"] != 0 else 0
    assert abs(stats["sharpe"] - expected) < 1e-10


def test_flat_prices_zero_volatility():
    flat = pd.Series([100.0, 100.0, 100.0, 100.0])
    r = calculate_returns(flat)
    assert annualize_volatility(r) == 0.0


def test_weekly_returns_fewer_than_daily():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.Series(range(100, 130), index=idx, dtype=float)
    daily = calculate_returns(prices, frequency="daily")
    weekly = calculate_returns(prices, frequency="weekly")
    assert len(weekly) < len(daily)
