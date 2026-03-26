import pandas as pd


def get_correlation_matrix(
    prices: pd.DataFrame,
    window: int = 60,
    regime_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation matrix.

    Two modes:
    - regime_dates=None: use the most recent `window` trading days.
    - regime_dates provided (pd.DatetimeIndex): filter data to those dates,
      ignore window, compute a static correlation across all matching dates.

    NaN values are dropped per pair before Pearson calculation.

    Returns square symmetric DataFrame.
    """
    if regime_dates is not None:
        matching = prices.index.intersection(regime_dates)
        subset = prices.loc[matching]
    else:
        subset = prices.tail(window)

    return subset.corr(method="pearson")
