import numpy as np
import pandas as pd

FREQ_PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}
RESAMPLE_RULES = {"daily": None, "weekly": "W", "monthly": "ME"}


def calculate_returns(prices: pd.Series, frequency: str = "daily") -> pd.Series:
    """
    Calculate simple percentage returns from a price series.
    frequency: 'daily' (no resampling), 'weekly', or 'monthly'.
    Resamples prices to the target frequency before computing returns.
    """
    rule = RESAMPLE_RULES.get(frequency)
    if rule:
        prices = prices.resample(rule).last().dropna()
    return prices.pct_change().dropna()


def annualize_return(returns: pd.Series, frequency: str = "daily") -> float:
    """Compound annualized return from a returns series."""
    n = FREQ_PERIODS[frequency]
    return float((1 + returns.mean()) ** n - 1)


def annualize_volatility(returns: pd.Series, frequency: str = "daily") -> float:
    """Annualized volatility (standard deviation of returns)."""
    n = FREQ_PERIODS[frequency]
    return float(returns.std() * np.sqrt(n))


def compute_stats(returns: pd.Series, frequency: str = "daily") -> dict:
    """
    Compute key return statistics.
    Returns dict with: mean_return, volatility, skewness, kurtosis, sharpe.
    Sharpe uses no risk-free rate adjustment (simplified).
    """
    vol = annualize_volatility(returns, frequency)
    ret = annualize_return(returns, frequency)
    return {
        "mean_return": ret,
        "volatility": vol,
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),  # excess kurtosis (normal = 0)
        "sharpe": ret / vol if vol != 0 else 0.0,
    }
