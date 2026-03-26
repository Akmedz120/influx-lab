import pytest
import pandas as pd
from modules.market_pulse.scoring import percentile_score


@pytest.fixture
def uniform_series():
    return pd.Series(range(0, 101))  # 0-100, makes percentile math predictable


def test_high_value_is_stressed(uniform_series):
    result = percentile_score(uniform_series, 90.0)
    assert result["color"] == "red"
    assert result["label"] == "Stressed"


def test_low_value_is_calm(uniform_series):
    result = percentile_score(uniform_series, 10.0)
    assert result["color"] == "green"
    assert result["label"] == "Calm"


def test_middle_value_is_neutral(uniform_series):
    result = percentile_score(uniform_series, 50.0)
    assert result["color"] == "yellow"
    assert result["label"] == "Neutral"


def test_invert_flips_green_and_red(uniform_series):
    # Without invert: 90th pct = red. With invert: 90th pct = green.
    result = percentile_score(uniform_series, 90.0, invert=True)
    assert result["color"] == "green"
    assert result["label"] == "Calm"


def test_empty_series_returns_neutral():
    result = percentile_score(pd.Series([], dtype=float), 5.0)
    assert result["color"] == "yellow"
    assert result["score"] == 50.0


def test_returns_required_keys(uniform_series):
    result = percentile_score(uniform_series, 50.0)
    assert all(k in result for k in ["score", "color", "label"])


def test_score_is_float(uniform_series):
    result = percentile_score(uniform_series, 50.0)
    assert isinstance(result["score"], float)
