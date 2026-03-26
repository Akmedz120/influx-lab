import pandas as pd
import yfinance as yf
from modules.data.cache import get_cached, set_cached
from config import FRED_API_KEY, CACHE_TTL_DAILY


def _yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around yfinance.download — isolated for mocking in tests."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        # Single ticker returns flat DataFrame
        data = data[["Close"]].rename(columns={"Close": tickers[0]})
    return data


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for one or more tickers.
    Returns DataFrame with ticker columns and DatetimeIndex.
    Results are cached locally for 24 hours.
    """
    key = f"prices_{'_'.join(sorted(tickers))}_{start}_{end}"
    cached = get_cached(key, ttl_hours=CACHE_TTL_DAILY)
    if cached is not None:
        return cached
    data = _yf_download(tickers, start, end)
    set_cached(key, data, ttl_hours=CACHE_TTL_DAILY)
    return data


def load_csv(file) -> pd.DataFrame:
    """
    Load a CSV file (file-like object or path) into a DataFrame.
    Raises ValueError if the file is empty or unparseable.
    """
    try:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("CSV file is empty")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty or has no columns")


def _fred_get_series(series_id: str, start: str, end: str) -> pd.Series:
    """Thin wrapper around fredapi — isolated for mocking in tests."""
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    return fred.get_series(series_id, observation_start=start, observation_end=end)


def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """
    Fetch a FRED economic data series by series ID.
    Returns a pd.Series with DatetimeIndex.
    Results cached for 24 hours.

    Common series IDs:
      FEDFUNDS     — Federal funds rate
      DGS2         — 2-year Treasury yield
      DGS10        — 10-year Treasury yield
      BAMLH0A0HYM2 — High yield credit spread
      M2SL         — M2 money supply
      UMCSENT      — University of Michigan consumer sentiment
      AAIIBULL     — AAII bullish sentiment
    """
    key = f"fred_{series_id}_{start}_{end}"
    cached = get_cached(key, ttl_hours=CACHE_TTL_DAILY)
    if cached is not None:
        return cached
    data = _fred_get_series(series_id, start, end)
    set_cached(key, data, ttl_hours=CACHE_TTL_DAILY)
    return data
