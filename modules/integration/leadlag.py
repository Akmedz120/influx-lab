import pandas as pd


def cross_correlate(series_a: pd.Series, series_b: pd.Series, max_lag: int = 60) -> pd.Series:
    """
    Compute Pearson correlation between series_a and series_b at lags from
    -max_lag to +max_lag.

    At lag k: corr(a[t], b[t+k]) — positive k means a leads b.

    Returns a pd.Series indexed by integer lag (-max_lag … +max_lag).
    """
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        b_shifted = series_b.shift(-lag)
        combined = pd.concat([series_a, b_shifted], axis=1).dropna()
        if len(combined) < 2:
            results[lag] = float("nan")
        else:
            results[lag] = combined.iloc[:, 0].corr(combined.iloc[:, 1])
    return pd.Series(results)


def interpret_lag(
    indicator_name: str,
    target_name: str,
    peak_lag: int,
    peak_corr: float,
) -> str:
    """
    Return a plain-English description of the lead-lag relationship.

    Positive peak_lag  → indicator leads target.
    Negative peak_lag  → indicator lags target.
    Zero peak_lag      → indicator moves with target (no lead/lag).
    """
    direction = "positively" if peak_corr >= 0 else "negatively"
    corr_str = f"{abs(peak_corr):.2f}"

    if peak_lag > 0:
        return (
            f"{indicator_name} leads {target_name} by {peak_lag} trading days "
            f"(peak correlation: {corr_str}, {direction} correlated)."
        )
    elif peak_lag < 0:
        return (
            f"{indicator_name} lags {target_name} by {abs(peak_lag)} trading days "
            f"(peak correlation: {corr_str}, {direction} correlated)."
        )
    else:
        return (
            f"{indicator_name} moves with {target_name} with no lead/lag "
            f"(peak correlation: {corr_str}, {direction} correlated)."
        )


def scan_all_vs_target(
    target: pd.Series,
    indicators: dict[str, pd.Series],
    max_lag: int = 60,
) -> pd.DataFrame:
    """
    Cross-correlate each indicator against the target and summarise results.

    Returns a DataFrame with columns:
        indicator_name, peak_correlation, peak_lag, interpretation

    Sorted descending by abs(peak_correlation).
    """
    rows = []
    for name, series in indicators.items():
        xcorr = cross_correlate(series, target, max_lag=max_lag)
        peak_lag = int(xcorr.abs().idxmax())
        peak_corr = float(xcorr[peak_lag])
        interpretation = interpret_lag(name, target.name or "target", peak_lag, peak_corr)
        rows.append({
            "indicator_name": name,
            "peak_correlation": peak_corr,
            "peak_lag": peak_lag,
            "interpretation": interpretation,
        })

    df = pd.DataFrame(rows, columns=["indicator_name", "peak_correlation", "peak_lag", "interpretation"])
    return df.sort_values("peak_correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
