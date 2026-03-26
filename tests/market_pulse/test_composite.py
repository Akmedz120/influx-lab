import pytest
import pandas as pd
from contextlib import ExitStack
from unittest.mock import patch


PATCH_TARGETS = [
    "modules.market_pulse.composite.get_vix",
    "modules.market_pulse.composite.get_hy_credit_spread",
    "modules.market_pulse.composite.get_dxy",
    "modules.market_pulse.composite.get_yield_curve",
    "modules.market_pulse.composite.get_gold_spy_ratio",
    "modules.market_pulse.composite.get_aaii_sentiment",
]


def make_indicator(current, series_values, invert=False):
    return {
        "series": pd.Series(list(series_values)),
        "current": float(current),
        "label": "Test",
        "unit": "x",
        "invert": invert,
    }


def mock_all(return_value):
    """Patch all 6 composite indicator functions with the same return value."""
    ctx = ExitStack()
    for target in PATCH_TARGETS:
        ctx.enter_context(patch(target, return_value=return_value))
    return ctx


def test_returns_required_keys():
    from modules.market_pulse.composite import compute_fear_greed_index
    ind = make_indicator(50, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert all(k in result for k in ["score", "label", "details"])


def test_score_is_float_between_0_and_100():
    from modules.market_pulse.composite import compute_fear_greed_index
    ind = make_indicator(50, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 100.0


def test_all_stressed_gives_high_score():
    from modules.market_pulse.composite import compute_fear_greed_index
    # current=99 against range(101) → ~99th pct → red
    ind = make_indicator(99, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert result["score"] > 75
    assert result["label"] in ["Fear", "Extreme Fear"]


def test_all_calm_gives_low_score():
    from modules.market_pulse.composite import compute_fear_greed_index
    # current=1 against range(101) → ~1st pct → green
    ind = make_indicator(1, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert result["score"] < 25
    assert result["label"] in ["Greed", "Extreme Greed"]


def test_details_has_one_entry_per_indicator():
    from modules.market_pulse.composite import compute_fear_greed_index
    ind = make_indicator(50, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert len(result["details"]) == 6


def test_failed_indicator_is_skipped_gracefully():
    from modules.market_pulse.composite import compute_fear_greed_index
    # 5 good, 1 raises — should still return a score from the 5
    ind = make_indicator(50, range(101))
    ctx = ExitStack()
    for i, target in enumerate(PATCH_TARGETS):
        if i == 0:
            ctx.enter_context(patch(target, side_effect=Exception("no data")))
        else:
            ctx.enter_context(patch(target, return_value=ind))
    with ctx:
        result = compute_fear_greed_index()
    assert len(result["details"]) == 5
    assert 0.0 <= result["score"] <= 100.0
