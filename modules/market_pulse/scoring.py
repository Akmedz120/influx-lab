import pandas as pd
from scipy import stats


def percentile_score(series: pd.Series, value: float, invert: bool = False) -> dict:
    """
    Score `value` against `series` historical data using percentile rank.

    Parameters:
        series : Historical time series (e.g., 3 years of daily data)
        value  : Current observed value to score
        invert : True for indicators where high value = good/calm
                 (e.g., consumer confidence, yield curve spread, AAII bull-bear)

    Returns dict:
        score  : 0–100 percentile rank (100 = most stressed)
        color  : 'green' (< 25th pct) | 'yellow' (25–75th) | 'red' (> 75th)
        label  : 'Calm' | 'Neutral' | 'Stressed'
    """
    clean = series.dropna()
    if len(clean) == 0:
        return {"score": 50.0, "color": "yellow", "label": "Neutral"}
    pct = float(stats.percentileofscore(clean, value, kind="rank"))
    if invert:
        pct = 100.0 - pct
    color = "green" if pct < 25 else "red" if pct > 75 else "yellow"
    label = "Calm" if pct < 25 else "Stressed" if pct > 75 else "Neutral"
    return {"score": round(pct, 1), "color": color, "label": label}
