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
