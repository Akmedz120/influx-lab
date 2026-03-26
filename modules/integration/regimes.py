import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


REGIME_ORDER = ["Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear"]

REGIME_COLORS = {
    "Extreme Greed": "#00CC96",
    "Greed":         "#7FBA00",
    "Neutral":       "#FFA500",
    "Fear":          "#FF6B35",
    "Extreme Fear":  "#EF553B",
}

_REGIME_BANDS = [
    (80, "Extreme Fear"),
    (60, "Fear"),
    (40, "Neutral"),
    (20, "Greed"),
    (0,  "Extreme Greed"),
]


def classify_regime(score: float) -> str:
    """Map a single F&G score (0–100) to a regime label."""
    for threshold, label in _REGIME_BANDS:
        if score >= threshold:
            return label
    return "Extreme Greed"


def get_regime_history(fg_scores: pd.Series) -> pd.Series:
    """Map a Series of F&G scores to regime labels. Preserves index."""
    return fg_scores.map(classify_regime)


def get_fear_greed_history() -> pd.Series:
    """
    Reconstruct the daily Fear & Greed composite score over the 3-year window.

    Fetches the same 6 indicator series used by compute_fear_greed_index().
    Scores each historical value against the FULL 3-year distribution
    (not rolling — acceptable for visualization, not for backtesting).
    Equal-weights available indicators per date.
    Excludes dates where fewer than 3 indicators have data.

    Returns pd.Series indexed by date, values 0–100.
    """
    from modules.market_pulse.indicators import (
        get_vix, get_hy_credit_spread, get_dxy,
        get_yield_curve, get_gold_spy_ratio, get_aaii_sentiment,
    )

    COMPOSITE_FNS = [
        get_vix, get_hy_credit_spread, get_dxy,
        get_yield_curve, get_gold_spy_ratio, get_aaii_sentiment,
    ]

    scored_series = []
    for fn in COMPOSITE_FNS:
        try:
            ind = fn()
            series = ind["series"].dropna()
            invert = ind.get("invert", False)
            scores = series.map(
                lambda v, s=series: float(scipy_stats.percentileofscore(s, v, kind="rank"))
            )
            if invert:
                scores = 100.0 - scores
            scored_series.append(scores.rename(ind["label"]))
        except Exception:
            pass

    if not scored_series:
        return pd.Series(dtype=float)

    combined = pd.concat(scored_series, axis=1)
    valid_mask = combined.notna().sum(axis=1) >= 3
    result = combined[valid_mask].mean(axis=1)
    return result.dropna().sort_index()


def get_regime_asset_stats(
    regime_history: pd.Series,
    asset_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each regime, compute the mean daily log return of each asset
    on dates classified as that regime.

    Returns DataFrame indexed by regime (REGIME_ORDER), columns = asset names.
    Cells with fewer than 5 observations are NaN.
    """
    rows = {}
    for regime in REGIME_ORDER:
        regime_dates = regime_history[regime_history == regime].index
        matching = asset_prices.index.intersection(regime_dates)
        row = {}
        for col in asset_prices.columns:
            prices_in = asset_prices[col].loc[matching].sort_index().dropna()
            if len(prices_in) < 2:
                row[col] = float("nan")
                continue
            log_returns = np.log(prices_in / prices_in.shift(1)).dropna()
            row[col] = float(log_returns.mean()) if len(log_returns) >= 5 else float("nan")
        rows[regime] = row

    return pd.DataFrame(rows).T.reindex(REGIME_ORDER)
