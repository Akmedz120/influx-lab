import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def volatility_regime(prices: pd.Series) -> pd.Series:
    """
    20-day realized vol percentile-ranked against trailing 252-day window.
    Returns score 0-100 (higher = more volatile than usual).
    NaN for first 252 dates (insufficient history).
    """
    log_returns = np.log(prices / prices.shift(1))
    vol_20d = log_returns.rolling(20).std() * np.sqrt(252)

    result = pd.Series(index=prices.index, dtype=float)
    for i in range(len(vol_20d)):
        if i < 252:
            result.iloc[i] = float("nan")
            continue
        window = vol_20d.iloc[i - 252 : i + 1].dropna()
        current = vol_20d.iloc[i]
        if pd.isna(current) or len(window) < 2:
            result.iloc[i] = float("nan")
        else:
            result.iloc[i] = float(
                scipy_stats.percentileofscore(window, current, kind="rank")
            )
    return result


def momentum_short(prices: pd.Series) -> pd.Series:
    """20-day log return. Positive = rising, negative = falling."""
    return np.log(prices / prices.shift(20))


def momentum_long(prices: pd.Series) -> pd.Series:
    """90-day log return. Positive = rising, negative = falling."""
    return np.log(prices / prices.shift(90))


def mean_reversion(prices: pd.Series) -> pd.Series:
    """
    Z-score: (price - 60d SMA) / 60d rolling std.
    Positive = above average (stretched up), negative = below average (oversold).
    NaN when std = 0 (constant prices).
    """
    sma = prices.rolling(60).mean()
    std = prices.rolling(60).std()
    return (prices - sma) / std


def trend_strength(prices: pd.Series) -> pd.Series:
    """
    Rolling R² of OLS fit on 60-day windows of log prices.
    0 = no trend, 1 = perfect linear trend. NaN for first 59 dates.
    """
    log_prices = np.log(prices.replace(0, float("nan")))
    result = pd.Series(index=prices.index, dtype=float)
    x = np.arange(60, dtype=float)
    for i in range(len(log_prices)):
        if i < 59:
            result.iloc[i] = float("nan")
            continue
        y = log_prices.iloc[i - 59 : i + 1].values
        if np.any(np.isnan(y)):
            result.iloc[i] = float("nan")
            continue
        _, _, r_value, _, _ = scipy_stats.linregress(x, y)
        result.iloc[i] = r_value ** 2
    return result


def macro_stress(prices: pd.Series) -> pd.Series:
    """
    Calls get_fear_greed_history() and reindexes to prices.index via forward-fill.
    The prices argument is used only to determine the output index — its values are ignored.
    Returns NaN for dates outside the 3-year history window.
    Score: 0 = extreme greed (calm), 100 = extreme fear (stressed).
    """
    from modules.integration.regimes import get_fear_greed_history
    fg = get_fear_greed_history()
    return fg.reindex(prices.index, method="ffill")


def volume_regime(volume: pd.Series | None) -> pd.Series:
    """
    20-day average volume percentile-ranked against trailing 252-day window.
    Returns score 0-100 (higher = unusually high volume).
    If volume is None or empty, returns empty Series of NaN.
    """
    if volume is None or len(volume) == 0:
        return pd.Series(dtype=float)

    vol_20d = volume.rolling(20).mean()
    result = pd.Series(index=volume.index, dtype=float)
    for i in range(len(vol_20d)):
        if i < 252:
            result.iloc[i] = float("nan")
            continue
        window = vol_20d.iloc[i - 252 : i + 1].dropna()
        current = vol_20d.iloc[i]
        if pd.isna(current) or len(window) < 2:
            result.iloc[i] = float("nan")
        else:
            result.iloc[i] = float(
                scipy_stats.percentileofscore(window, current, kind="rank")
            )
    return result


def fiftytwo_week_position(prices: pd.Series) -> pd.Series:
    """
    Where is the current price in its 52-week range?
    0 = at the 52-week low, 100 = at the 52-week high.
    Returns 50 when high == low (flat price, no range).
    """
    rolling_min = prices.rolling(252).min()
    rolling_max = prices.rolling(252).max()
    denom = rolling_max - rolling_min
    result = (prices - rolling_min) / denom * 100
    return result.where(denom != 0, other=50.0)


def relative_strength(prices: pd.Series, benchmark: pd.Series) -> pd.Series:
    """
    Asset 20-day log return minus benchmark 20-day log return, aligned on common dates.
    Positive = outperforming benchmark, negative = underperforming.
    """
    asset_mom = np.log(prices / prices.shift(20))
    bench_mom = np.log(benchmark / benchmark.shift(20))
    combined  = pd.concat([asset_mom, bench_mom], axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    rs = combined.iloc[:, 0] - combined.iloc[:, 1]
    return rs.reindex(prices.index)


def _score_to_label(score: float, low: float = 33, high: float = 67) -> str:
    """Convert a 0–100 score to a low/mid/high label. Returns NaN for NaN input."""
    if pd.isna(score):
        return float("nan")
    if score < low:
        return "low"
    elif score > high:
        return "high"
    return "mid"
