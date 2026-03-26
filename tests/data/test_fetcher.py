import pytest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch
from modules.data.fetcher import fetch_prices, load_csv


@pytest.fixture(autouse=True)
def patch_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.data.cache.CACHE_DIR", tmp_path)


def make_price_df(tickers, n=5):
    idx = pd.date_range("2024-01-01", periods=n)
    data = {t: np.linspace(100, 110, n) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_fetch_prices_returns_dataframe_with_correct_columns():
    mock_df = make_price_df(["AAPL"])
    with patch("modules.data.fetcher._yf_download", return_value=mock_df):
        result = fetch_prices(["AAPL"], "2024-01-01", "2024-12-31")
    assert isinstance(result, pd.DataFrame)
    assert "AAPL" in result.columns
    assert len(result) == 5


def test_fetch_prices_multi_ticker_returns_all_columns():
    mock_df = make_price_df(["SPY", "QQQ"])
    with patch("modules.data.fetcher._yf_download", return_value=mock_df):
        result = fetch_prices(["SPY", "QQQ"], "2024-01-01", "2024-12-31")
    assert "SPY" in result.columns
    assert "QQQ" in result.columns


def test_fetch_prices_uses_cache_on_second_call():
    mock_df = make_price_df(["QQQ"])
    with patch("modules.data.fetcher._yf_download", return_value=mock_df) as mock_fn:
        fetch_prices(["QQQ"], "2024-01-01", "2024-06-01")
        fetch_prices(["QQQ"], "2024-01-01", "2024-06-01")
    assert mock_fn.call_count == 1


def test_load_csv_returns_dataframe():
    csv_content = "date,price\n2024-01-01,100\n2024-01-02,101\n"
    file = io.StringIO(csv_content)
    result = load_csv(file)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "price" in result.columns


def test_load_csv_empty_raises():
    file = io.StringIO("")
    with pytest.raises(Exception):
        load_csv(file)


from modules.data.fetcher import fetch_fred


def test_fetch_fred_returns_series():
    mock_series = pd.Series([5.25, 5.25, 5.0], index=pd.date_range("2024-01-01", periods=3))
    with patch("modules.data.fetcher._fred_get_series", return_value=mock_series):
        result = fetch_fred("FEDFUNDS", "2024-01-01", "2024-03-01")
    assert isinstance(result, pd.Series)
    assert len(result) == 3


def test_fetch_fred_uses_cache():
    mock_series = pd.Series([5.25], index=pd.date_range("2024-01-01", periods=1))
    with patch("modules.data.fetcher._fred_get_series", return_value=mock_series) as mock_fn:
        fetch_fred("DGS10", "2024-01-01", "2024-01-01")
        fetch_fred("DGS10", "2024-01-01", "2024-01-01")
    assert mock_fn.call_count == 1
