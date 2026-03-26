import pandas as pd
from modules.features.signals import (
    volatility_regime, momentum_short, momentum_long,
    mean_reversion, trend_strength, macro_stress,
    volume_regime, fiftytwo_week_position, relative_strength,
    _score_to_label,
)


def build(
    prices: pd.Series,
    volume: pd.Series | None = None,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute all 9 signals for the given price series.

    Parameters
    ----------
    prices    : daily close prices, any ticker
    volume    : daily volume for the same ticker (optional)
    benchmark : benchmark price series for relative strength (optional)

    Returns
    -------
    DataFrame indexed by prices.index with columns:
        vol_regime_score, vol_regime_label,
        momentum_short, momentum_long,
        mean_reversion, trend_strength,
        macro_stress,
        volume_regime_score, volume_regime_label,
        fiftytwo_week_pos,
        relative_strength
    """
    df = pd.DataFrame(index=prices.index)

    # Volatility regime
    vol_score = volatility_regime(prices)
    df["vol_regime_score"] = vol_score
    df["vol_regime_label"] = vol_score.map(_score_to_label)

    # Momentum
    df["momentum_short"] = momentum_short(prices)
    df["momentum_long"]  = momentum_long(prices)

    # Mean reversion
    df["mean_reversion"] = mean_reversion(prices)

    # Trend strength
    df["trend_strength"] = trend_strength(prices)

    # Macro stress (may fail if data unavailable)
    try:
        df["macro_stress"] = macro_stress(prices)
    except Exception:
        df["macro_stress"] = float("nan")

    # Volume regime
    vol_reg = volume_regime(volume)
    if vol_reg.empty:
        df["volume_regime_score"] = float("nan")
        df["volume_regime_label"] = float("nan")
    else:
        aligned = vol_reg.reindex(prices.index)
        df["volume_regime_score"] = aligned
        df["volume_regime_label"] = aligned.map(_score_to_label)

    # 52-week position
    df["fiftytwo_week_pos"] = fiftytwo_week_position(prices)

    # Relative strength
    if benchmark is not None:
        df["relative_strength"] = relative_strength(prices, benchmark)
    else:
        df["relative_strength"] = float("nan")

    return df
