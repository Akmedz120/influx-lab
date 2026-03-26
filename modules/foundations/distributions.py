import numpy as np
import pandas as pd
from scipy import stats


def fit_normal(returns: pd.Series) -> tuple[float, float]:
    """Fit a normal distribution to the returns. Returns (mu, sigma) via MLE."""
    mu, sigma = stats.norm.fit(returns)
    return float(mu), float(sigma)


def normal_pdf_range(returns: pd.Series, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate x/y values for a normal distribution fitted to returns.
    Use to overlay a normal curve on a histogram for visual comparison.
    """
    mu, sigma = fit_normal(returns)
    x = np.linspace(returns.min(), returns.max(), n_points)
    y = stats.norm.pdf(x, mu, sigma)
    return x, y


def normality_test(returns: pd.Series) -> dict:
    """
    Shapiro-Wilk test for normality.
    Returns dict with statistic, p_value, is_normal.
    p_value > 0.05 = cannot reject normality.
    Samples up to 5000 observations (Shapiro-Wilk limit).
    """
    sample = returns.sample(min(len(returns), 5000), random_state=42)
    stat, p_value = stats.shapiro(sample)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05),
    }
