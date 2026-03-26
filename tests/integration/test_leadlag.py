import numpy as np
import pandas as pd


def _make_lagged_pair(n=300, lag=15, seed=42):
    """Two series where series_a leads series_b by `lag` trading days."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(seed)
    a_vals = np.random.normal(0, 1, n)
    b_vals = np.roll(a_vals, lag)
    b_vals[:lag] = 0.0
    a = pd.Series(a_vals, index=idx, name="a")
    b = pd.Series(b_vals + np.random.normal(0, 0.05, n), index=idx, name="b")
    return a, b


# ─── cross_correlate ──────────────────────────────────────────────────────────

def test_cross_correlate_returns_correct_index():
    from modules.integration.leadlag import cross_correlate
    a, b = _make_lagged_pair()
    result = cross_correlate(a, b, max_lag=30)
    assert list(result.index) == list(range(-30, 31))


def test_cross_correlate_lag0_matches_pearson():
    from modules.integration.leadlag import cross_correlate
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(99)
    a = pd.Series(np.random.normal(0, 1, 100), index=idx)
    b = pd.Series(np.random.normal(0, 1, 100), index=idx)
    result = cross_correlate(a, b, max_lag=10)
    expected = float(a.corr(b))
    assert abs(result[0] - expected) < 1e-9


def test_cross_correlate_detects_positive_lag():
    from modules.integration.leadlag import cross_correlate
    a, b = _make_lagged_pair(n=300, lag=15)
    result = cross_correlate(a, b, max_lag=30)
    peak_lag = int(result.abs().idxmax())
    assert 10 <= peak_lag <= 20


# ─── interpret_lag ────────────────────────────────────────────────────────────

def test_interpret_lag_positive():
    from modules.integration.leadlag import interpret_lag
    result = interpret_lag("Yield Curve", "VIX", peak_lag=45, peak_corr=0.71)
    assert "leads" in result
    assert "45" in result


def test_interpret_lag_negative():
    from modules.integration.leadlag import interpret_lag
    result = interpret_lag("XLK", "SPY", peak_lag=-3, peak_corr=0.88)
    assert "lags" in result
    assert "3" in result


def test_interpret_lag_zero():
    from modules.integration.leadlag import interpret_lag
    result = interpret_lag("VIX", "SPY", peak_lag=0, peak_corr=-0.75)
    assert "no lead" in result.lower() or "moves with" in result.lower()


# ─── scan_all_vs_target ───────────────────────────────────────────────────────

def test_scan_all_vs_target_returns_dataframe():
    from modules.integration.leadlag import scan_all_vs_target
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)
    target = pd.Series(np.random.normal(0, 1, 200), index=idx, name="VIX")
    indicators = {
        "DXY": pd.Series(np.random.normal(0, 1, 200), index=idx),
        "SPY": pd.Series(np.random.normal(0, 1, 200), index=idx),
    }
    result = scan_all_vs_target(target, indicators, max_lag=10)
    assert isinstance(result, pd.DataFrame)


def test_scan_all_vs_target_has_required_columns():
    from modules.integration.leadlag import scan_all_vs_target
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)
    target = pd.Series(np.random.normal(0, 1, 200), index=idx, name="VIX")
    indicators = {"DXY": pd.Series(np.random.normal(0, 1, 200), index=idx)}
    result = scan_all_vs_target(target, indicators, max_lag=10)
    assert all(c in result.columns for c in ["indicator_name", "peak_correlation", "peak_lag", "interpretation"])


def test_scan_all_vs_target_sorted_by_abs_peak_correlation():
    from modules.integration.leadlag import scan_all_vs_target
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    np.random.seed(7)
    target = pd.Series(np.random.normal(0, 1, 300), index=idx, name="VIX")
    strong = target * 0.9 + pd.Series(np.random.normal(0, 0.1, 300), index=idx)
    weak = pd.Series(np.random.normal(0, 1, 300), index=idx)
    indicators = {"strong": strong, "weak": weak}
    result = scan_all_vs_target(target, indicators, max_lag=10)
    assert result.iloc[0]["indicator_name"] == "strong"
