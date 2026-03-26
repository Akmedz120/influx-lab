import numpy as np
import pandas as pd


def _make_price_df(n=90, seed=42):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(seed)
    return pd.DataFrame({
        "VIX": np.random.normal(20, 3, n),
        "DXY": np.random.normal(100, 2, n),
        "SPY": np.random.normal(450, 10, n),
    }, index=idx)


def test_get_correlation_matrix_is_square():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    assert result.shape[0] == result.shape[1] == 3


def test_get_correlation_matrix_diagonal_is_one():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    for col in result.columns:
        assert abs(result.loc[col, col] - 1.0) < 1e-9


def test_get_correlation_matrix_values_bounded():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    assert (result.fillna(0).values >= -1.0 - 1e-9).all()
    assert (result.fillna(0).values <= 1.0 + 1e-9).all()


def test_get_correlation_matrix_is_symmetric():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    pd.testing.assert_frame_equal(result, result.T)


def test_get_correlation_matrix_window_uses_tail():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df(n=90)
    result_30 = get_correlation_matrix(prices, window=30)
    result_60 = get_correlation_matrix(prices, window=60)
    assert not result_30.equals(result_60)


def test_get_correlation_matrix_regime_filter_uses_given_dates():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df(n=90)
    regime_dates = prices.index[:30]
    result_regime = get_correlation_matrix(prices, regime_dates=regime_dates)
    result_first30 = get_correlation_matrix(prices.iloc[:30])
    pd.testing.assert_frame_equal(result_regime, result_first30)


def test_get_correlation_matrix_handles_nan():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    prices.iloc[10:20, 0] = np.nan
    result = get_correlation_matrix(prices)
    assert abs(result.loc["VIX", "VIX"] - 1.0) < 1e-9
