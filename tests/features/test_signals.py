import numpy as np
import pandas as pd


def _make_prices(n=300, seed=42):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(seed)
    returns = np.random.normal(0, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + returns), index=idx)


def _make_rising(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.linspace(100, 200, n), index=idx)


def _make_falling(n=300):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.linspace(200, 100, n), index=idx)


# ─── volatility_regime ────────────────────────────────────────────────────────

def test_volatility_regime_returns_series():
    from modules.features.signals import volatility_regime
    prices = _make_prices()
    result = volatility_regime(prices)
    assert isinstance(result, pd.Series)
    assert result.index.equals(prices.index)


def test_volatility_regime_labels_are_valid():
    from modules.features.signals import volatility_regime, _score_to_label
    prices = _make_prices(n=500)
    scores = volatility_regime(prices)
    labels = scores.dropna().map(_score_to_label)
    assert labels.isin({"low", "mid", "high"}).all()


def test_volatility_regime_high_vol_scores_high():
    from modules.features.signals import volatility_regime
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(0)
    returns = np.concatenate([
        np.random.normal(0, 0.005, 400),  # calm period
        np.random.normal(0, 0.05, 100),   # high vol burst
    ])
    prices = pd.Series(100 * np.cumprod(1 + returns), index=idx)
    result = volatility_regime(prices)
    assert result.dropna().iloc[-1] > 67


# ─── momentum_short ───────────────────────────────────────────────────────────

def test_momentum_short_positive_for_rising_prices():
    from modules.features.signals import momentum_short
    result = momentum_short(_make_rising())
    assert result.dropna().iloc[-1] > 0


def test_momentum_short_negative_for_falling_prices():
    from modules.features.signals import momentum_short
    result = momentum_short(_make_falling())
    assert result.dropna().iloc[-1] < 0


# ─── momentum_long ────────────────────────────────────────────────────────────

def test_momentum_long_uses_90d_window():
    from modules.features.signals import momentum_long
    prices = _make_rising(n=200)
    result = momentum_long(prices)
    i = 150
    expected = float(np.log(prices.iloc[i] / prices.iloc[i - 90]))
    assert abs(result.iloc[i] - expected) < 1e-9


# ─── mean_reversion ───────────────────────────────────────────────────────────

def test_mean_reversion_zero_at_moving_average():
    from modules.features.signals import mean_reversion
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(7)
    # Stationary series — mean of z-scores should be near 0 by definition
    prices = pd.Series(100 + np.random.normal(0, 1, 100), index=idx)
    result = mean_reversion(prices)
    assert abs(result.dropna().mean()) < 0.5


def test_mean_reversion_positive_above_sma():
    from modules.features.signals import mean_reversion
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    prices = pd.Series(np.concatenate([
        100 * np.ones(100),
        np.linspace(100, 200, 100),
    ]), index=idx)
    result = mean_reversion(prices)
    assert result.dropna().iloc[-1] > 0


# ─── trend_strength ───────────────────────────────────────────────────────────

def test_trend_strength_high_for_linear_trend():
    from modules.features.signals import trend_strength
    result = trend_strength(_make_rising(n=200))
    assert result.dropna().iloc[-1] > 0.95


def test_trend_strength_low_for_random_walk():
    from modules.features.signals import trend_strength
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(99)
    returns = np.random.normal(0, 0.01, 300)
    prices = pd.Series(100 * np.cumprod(1 + returns), index=idx)
    result = trend_strength(prices)
    assert result.dropna().mean() < 0.5


# ─── fiftytwo_week_position ───────────────────────────────────────────────────

def test_fiftytwo_week_position_at_high():
    from modules.features.signals import fiftytwo_week_position
    result = fiftytwo_week_position(_make_rising(n=300))
    assert abs(result.dropna().iloc[-1] - 100.0) < 1e-6


def test_fiftytwo_week_position_at_low():
    from modules.features.signals import fiftytwo_week_position
    result = fiftytwo_week_position(_make_falling(n=300))
    assert abs(result.dropna().iloc[-1] - 0.0) < 1e-6


# ─── relative_strength ────────────────────────────────────────────────────────

def test_relative_strength_zero_when_equal():
    from modules.features.signals import relative_strength
    prices = _make_rising()
    result = relative_strength(prices, prices)
    assert (result.dropna().abs() < 1e-9).all()


def test_relative_strength_positive_when_outperforms():
    from modules.features.signals import relative_strength
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    asset     = pd.Series(np.linspace(100, 200, 200), index=idx)
    benchmark = pd.Series(np.linspace(100, 110, 200), index=idx)
    result = relative_strength(asset, benchmark)
    assert result.dropna().iloc[-1] > 0


# ─── volume_regime ────────────────────────────────────────────────────────────

def test_volume_regime_nan_when_no_volume():
    from modules.features.signals import volume_regime
    result = volume_regime(None)
    assert isinstance(result, pd.Series)
    assert len(result) == 0 or result.isna().all()
