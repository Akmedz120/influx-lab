# Phase 4 — Features Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature engineering module that computes 9 market signals for any ticker, displays them in a visual dashboard (radar chart, calendar heatmap, divergence flags), and exports a clean date × feature table as CSV for future ML use.

**Architecture:** A `modules/features/` package with `signals.py` (9 pure signal functions) and `feature_table.py` (assembles all signals into a DataFrame). The page `4_Features.py` fetches data, builds the feature table, and renders three sections: Signal Snapshot, Signal History, Feature Table.

**Tech Stack:** Python 3.13, pandas, numpy, scipy, plotly, streamlit, yfinance (already installed). No new dependencies.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `modules/features/__init__.py` | Create (empty) | Package marker |
| `modules/features/signals.py` | Create | 9 signal functions + `_score_to_label` helper |
| `modules/features/feature_table.py` | Create | `build()` assembles all signals into date × feature DataFrame |
| `tests/features/__init__.py` | Create (empty) | Test package marker |
| `tests/features/test_signals.py` | Create | 15 unit tests for signal functions |
| `tests/features/test_feature_table.py` | Create | 3 unit tests for `build()` |
| `pages/4_Features.py` | Modify | Replace stub with full UI |

---

## Task 1: Scaffold

**Files:**
- Create: `modules/features/__init__.py`
- Create: `tests/features/__init__.py`

- [ ] **Step 1: Create both empty `__init__.py` files**

```bash
touch /Users/akhilm/Claude-Projects/influx-lab/modules/features/__init__.py
touch /Users/akhilm/Claude-Projects/influx-lab/tests/features/__init__.py
```

- [ ] **Step 2: Verify they exist**

```bash
ls /Users/akhilm/Claude-Projects/influx-lab/modules/features/
ls /Users/akhilm/Claude-Projects/influx-lab/tests/features/
```

Expected: each directory shows `__init__.py`

- [ ] **Step 3: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/features/__init__.py tests/features/__init__.py && git commit -m "feat: scaffold features module"
```

---

## Task 2: signals.py

**Files:**
- Create: `modules/features/signals.py`
- Create: `tests/features/test_signals.py`

### Step 1: Write failing tests

- [ ] **Step 1: Create `tests/features/test_signals.py`**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/features/test_signals.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'modules.features.signals'`

### Step 3: Write implementation

- [ ] **Step 3: Create `modules/features/signals.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/features/test_signals.py -v 2>&1 | tail -20
```

Expected: 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/features/signals.py tests/features/test_signals.py && git commit -m "feat: add 9 market signal functions"
```

---

## Task 3: feature_table.py

**Files:**
- Create: `modules/features/feature_table.py`
- Create: `tests/features/test_feature_table.py`

### Step 1: Write failing tests

- [ ] **Step 1: Create `tests/features/test_feature_table.py`**

```python
import numpy as np
import pandas as pd
from unittest.mock import patch


REQUIRED_COLUMNS = [
    "vol_regime_score", "vol_regime_label",
    "momentum_short", "momentum_long",
    "mean_reversion", "trend_strength",
    "macro_stress",
    "volume_regime_score", "volume_regime_label",
    "fiftytwo_week_pos",
    "relative_strength",
]


def _make_prices(n=300, seed=42):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    np.random.seed(seed)
    returns = np.random.normal(0, 0.01, n)
    return pd.Series(100 * np.cumprod(1 + returns), index=idx)


def _patch_fg(prices):
    """Context manager: patch get_fear_greed_history to return a flat 50.0 series."""
    return patch(
        "modules.integration.regimes.get_fear_greed_history",
        return_value=pd.Series(50.0, index=prices.index),
    )


def test_feature_table_has_required_columns():
    from modules.features.feature_table import build
    prices = _make_prices()
    with _patch_fg(prices):
        result = build(prices)
    assert all(c in result.columns for c in REQUIRED_COLUMNS)


def test_feature_table_index_matches_prices():
    from modules.features.feature_table import build
    prices = _make_prices()
    with _patch_fg(prices):
        result = build(prices)
    assert result.index.equals(prices.index)


def test_feature_table_handles_missing_benchmark():
    from modules.features.feature_table import build
    prices = _make_prices()
    with _patch_fg(prices):
        result = build(prices, benchmark=None)
    assert result["relative_strength"].isna().all()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/features/test_feature_table.py -v 2>&1 | tail -8
```

Expected: `ModuleNotFoundError: No module named 'modules.features.feature_table'`

### Step 3: Write implementation

- [ ] **Step 3: Create `modules/features/feature_table.py`**

```python
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
    benchmark : benchmark price series for relative strength (optional, default None → NaN column)

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

    # Macro stress (may fail if FRED/yfinance data unavailable)
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/features/ -v 2>&1 | tail -25
```

Expected: 18 tests PASS (15 signals + 3 feature_table).

- [ ] **Step 5: Run full suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -q 2>&1 | tail -5
```

Expected: 130 passed (112 existing + 18 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/features/feature_table.py tests/features/test_feature_table.py && git commit -m "feat: add feature table builder"
```

---

## Task 4: Integration Page UI (`pages/4_Features.py`)

**Files:**
- Modify: `pages/4_Features.py`

No unit tests — UI layer only.

**Before writing:** confirm `modules/features/signals.py` and `feature_table.py` exist and tests pass.

- [ ] **Step 1: Replace `pages/4_Features.py`**

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

from modules.features.feature_table import build
from modules.data.fetcher import fetch_prices

st.set_page_config(page_title="Features", layout="wide")
st.title("Features")
st.caption(
    "Market signals computed for any ticker — volatility state, momentum, trend strength, "
    "and more. A live fingerprint of where an asset stands right now, plus a downloadable "
    "feature table for future model building."
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

SIGNAL_LABELS = {
    "vol_regime_score":   "Volatility Regime",
    "momentum_short":     "Momentum (20d)",
    "momentum_long":      "Momentum (90d)",
    "mean_reversion":     "Mean Reversion",
    "trend_strength":     "Trend Strength",
    "macro_stress":       "Macro Stress",
    "volume_regime_score":"Volume Regime",
    "fiftytwo_week_pos":  "52-Week Position",
    "relative_strength":  "Relative Strength",
}

SIGNAL_EXPLANATIONS = {
    "vol_regime_score":    "How volatile is this asset vs its own history? High = unusually choppy.",
    "momentum_short":      "20-day price momentum. Positive = rising trend, negative = falling.",
    "momentum_long":       "90-day price momentum. Slower signal — shows the bigger trend direction.",
    "mean_reversion":      "How far is price from its 60-day average? High z-score = stretched up.",
    "trend_strength":      "How consistent is the price direction? Near 1 = strong steady trend.",
    "macro_stress":        "Macro Fear & Greed score (0 = calm, 100 = stressed). From Phase 3.",
    "volume_regime_score": "Is trading volume high or low vs recent history?",
    "fiftytwo_week_pos":   "Where is price in its 52-week range? 0 = near lows, 100 = near highs.",
    "relative_strength":   "20-day return vs SPY. Positive = outperforming the market.",
}

# Normalization bounds for radar chart (signed signals mapped to 0–1 with 0.5 = zero)
_RADAR_NORM = {
    "vol_regime_score":    ("bounded",  0, 100),
    "momentum_short":      ("signed",  -0.15, 0.15),
    "momentum_long":       ("signed",  -0.15, 0.15),
    "mean_reversion":      ("signed",  -3.0, 3.0),
    "trend_strength":      ("bounded",  0, 1),
    "macro_stress":        ("bounded",  0, 100),
    "volume_regime_score": ("bounded",  0, 100),
    "fiftytwo_week_pos":   ("bounded",  0, 100),
    "relative_strength":   ("signed",  -3.0, 3.0),
}

RADAR_SIGNALS = list(_RADAR_NORM.keys())


def _normalize_for_radar(value: float, signal: str) -> float:
    """Map signal value to [0, 1] for radar chart display."""
    if pd.isna(value):
        return 0.5
    mode, lo, hi = _RADAR_NORM[signal]
    if mode == "bounded":
        return float(np.clip((value - lo) / (hi - lo), 0, 1))
    else:  # signed: 0.5 = zero
        clipped = float(np.clip(value, lo, hi))
        return (clipped - lo) / (hi - lo)


def _badge_color(signal: str, value: float) -> str:
    """Green = calm/good, amber = mid, red = stressed/extreme."""
    if pd.isna(value):
        return "#888888"
    # Signals where high = stressed
    stressed_high = {"vol_regime_score", "macro_stress"}
    # Signals where high = good
    good_high = {"fiftytwo_week_pos", "trend_strength", "momentum_short",
                 "momentum_long", "relative_strength"}
    # Bounded signals
    if signal in stressed_high:
        norm = _normalize_for_radar(value, signal)
        if norm > 0.67: return "#EF553B"
        if norm > 0.33: return "#FFA500"
        return "#00CC96"
    if signal in good_high:
        norm = _normalize_for_radar(value, signal)
        if norm > 0.67: return "#00CC96"
        if norm > 0.33: return "#FFA500"
        return "#EF553B"
    # Volume and mean reversion: extremes (either direction) = amber
    norm = _normalize_for_radar(value, signal)
    if 0.33 <= norm <= 0.67: return "#00CC96"
    return "#FFA500"


def _check_divergences(row: pd.Series) -> list[str]:
    """Return list of divergence messages. Empty = no divergences."""
    msgs = []
    vol_label = row.get("vol_regime_label", "mid")
    ms = row.get("momentum_short", 0)
    ml = row.get("momentum_long", 0)
    macro = row.get("macro_stress", 50)

    if vol_label == "high" and not pd.isna(ms) and ms > 0 and not pd.isna(ml) and ml > 0:
        msgs.append(
            "Volatility regime is **high** but momentum is **positive** — "
            "stress and price trend are pointing in opposite directions."
        )
    if not pd.isna(macro) and macro > 60 and not pd.isna(ms) and ms > 0:
        msgs.append(
            "Macro stress is **elevated** (score > 60) but short-term momentum is **positive** — "
            "macro environment and price action disagree."
        )
    if not pd.isna(ms) and not pd.isna(ml) and ((ms > 0) != (ml > 0)):
        msgs.append(
            "Short-term (20d) and long-term (90d) momentum are pointing in **opposite directions** — "
            "trend may be reversing."
        )
    return msgs


@st.cache_data(ttl=3600)
def _fetch_data(ticker: str):
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=1200)).strftime("%Y-%m-%d")
    try:
        prices_df = fetch_prices([ticker], start, end)
        prices = prices_df[ticker].dropna() if ticker in prices_df.columns else pd.Series(dtype=float)
    except Exception:
        prices = pd.Series(dtype=float)
    # Fetch volume directly via yfinance (not in cached fetcher)
    volume = None
    try:
        hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        if "Volume" in hist.columns:
            volume = hist["Volume"].dropna()
            if volume.index.tz is not None:
                volume.index = volume.index.tz_convert(None)
    except Exception:
        pass
    # Fetch SPY as benchmark
    benchmark = None
    try:
        spy_df = fetch_prices(["SPY"], start, end)
        if "SPY" in spy_df.columns:
            benchmark = spy_df["SPY"].dropna()
    except Exception:
        pass
    return prices, volume, benchmark


# ─── Ticker Input ──────────────────────────────────────────────────────────────

ticker_input = st.text_input("Ticker", value="SPY", placeholder="e.g. SPY, AAPL, GLD, ^VIX").strip().upper()

if not ticker_input:
    st.stop()

prices, volume, benchmark = _fetch_data(ticker_input)

if prices.empty:
    st.error(f"Could not load price data for **{ticker_input}**. Check the ticker and try again.")
    st.stop()

# Build feature table
try:
    ft = build(prices, volume=volume, benchmark=benchmark)
except Exception as e:
    st.error(f"Could not build features: {e}")
    st.stop()

latest = ft.iloc[-1]

st.divider()

# ─── Section 1: Signal Snapshot ───────────────────────────────────────────────

st.header("Signal Snapshot")
st.caption(f"Current signal state for **{ticker_input}** as of {ft.index[-1].strftime('%Y-%m-%d')}.")

# Divergence flag
divs = _check_divergences(latest)
if divs:
    for msg in divs:
        st.warning(f"Divergence: {msg}")

# Radar chart
radar_vals = [_normalize_for_radar(latest.get(s, float("nan")), s) for s in RADAR_SIGNALS]
radar_labels = [SIGNAL_LABELS[s] for s in RADAR_SIGNALS]
radar_vals_closed = radar_vals + [radar_vals[0]]
radar_labels_closed = radar_labels + [radar_labels[0]]

fig_radar = go.Figure(go.Scatterpolar(
    r=radar_vals_closed,
    theta=radar_labels_closed,
    fill="toself",
    fillcolor="rgba(0, 150, 200, 0.15)",
    line=dict(color="rgba(0, 150, 200, 0.8)", width=2),
    hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75], ticktext=["", "", ""]),
        angularaxis=dict(tickfont=dict(size=11)),
    ),
    height=420,
    margin=dict(l=60, r=60, t=40, b=40),
    showlegend=False,
)
col_radar, col_badges = st.columns([1, 1])

with col_radar:
    st.plotly_chart(fig_radar, use_container_width=True)

with col_badges:
    st.markdown("**Signal Breakdown**")
    for sig, label in SIGNAL_LABELS.items():
        val = latest.get(sig, float("nan"))
        color = _badge_color(sig, val)
        val_str = f"{val:.3f}" if not pd.isna(val) else "—"
        st.markdown(
            f"<div style='padding:6px 10px;margin:4px 0;background:{color}22;"
            f"border-left:3px solid {color};border-radius:3px'>"
            f"<b>{label}</b>: {val_str}<br>"
            f"<small style='color:#888'>{SIGNAL_EXPLANATIONS.get(sig, '')}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ─── Section 2: Signal History ────────────────────────────────────────────────

st.header("Signal History")
st.caption("How any signal has evolved over time. The calendar heatmap shows patterns across a full year.")

all_signal_names = list(SIGNAL_LABELS.values())
all_signal_keys  = list(SIGNAL_LABELS.keys())
selected_label   = st.selectbox("Signal", all_signal_names, key="sig_hist")
selected_key     = all_signal_keys[all_signal_names.index(selected_label)]

sig_series = ft[selected_key].dropna()
sig_2y     = sig_series[sig_series.index >= (sig_series.index[-1] - pd.DateOffset(years=2))]

# Line chart with regime band shading for categorical signals
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=sig_2y.index, y=sig_2y.values,
    mode="lines",
    line=dict(color="#1E90FF", width=1.5),
    name=selected_label,
))
# Shade bands for vol_regime_score and volume_regime_score
if selected_key in ("vol_regime_score", "volume_regime_score", "macro_stress", "fiftytwo_week_pos"):
    fig_line.add_hrect(y0=0,  y1=33,  fillcolor="#00CC96", opacity=0.08, line_width=0)
    fig_line.add_hrect(y0=33, y1=67,  fillcolor="#FFA500", opacity=0.08, line_width=0)
    fig_line.add_hrect(y0=67, y1=100, fillcolor="#EF553B", opacity=0.08, line_width=0)
fig_line.update_layout(
    height=280,
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis_title=None,
    yaxis_title=selected_label,
    showlegend=False,
)
st.plotly_chart(fig_line, use_container_width=True)

# Calendar heatmap — last 1 year
sig_1y = sig_series[sig_series.index >= (sig_series.index[-1] - pd.DateOffset(years=1))]
if not sig_1y.empty:
    # Build 7 (day-of-week) × 53 (week-of-year) grid
    dow    = sig_1y.index.dayofweek   # Mon=0 … Sun=6
    week   = sig_1y.index.isocalendar().week.astype(int)
    year   = sig_1y.index.year

    # Normalize week numbers to 0–52 range for display
    min_week = week.min()
    col_idx  = week - min_week

    n_weeks  = int(col_idx.max()) + 1
    grid     = np.full((7, n_weeks), float("nan"))
    for d, w, v in zip(dow, col_idx, sig_1y.values):
        grid[d, w] = v

    fig_cal = go.Figure(go.Heatmap(
        z=grid,
        colorscale="RdYlGn_r" if selected_key in ("vol_regime_score", "macro_stress") else "RdYlGn",
        showscale=True,
        hoverongaps=False,
        xgap=2,
        ygap=2,
    ))
    fig_cal.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(
            tickvals=list(range(7)),
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            autorange="reversed",
        ),
        xaxis=dict(visible=False),
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    st.caption("Each square = one trading day. Color intensity = signal strength over the past year.")

st.divider()

# ─── Section 3: Feature Table ─────────────────────────────────────────────────

st.header("Feature Table")
st.caption(
    "Full date × feature matrix — the ML-ready output. "
    "Download as CSV to use in models, backtests, or further analysis."
)

display_ft = ft.sort_index(ascending=False).copy()
# Format for display
for col in ["momentum_short", "momentum_long", "mean_reversion", "relative_strength"]:
    if col in display_ft.columns:
        display_ft[col] = display_ft[col].map(lambda v: f"{v:.4f}" if not pd.isna(v) else "—")
for col in ["vol_regime_score", "macro_stress", "volume_regime_score", "fiftytwo_week_pos"]:
    if col in display_ft.columns:
        display_ft[col] = display_ft[col].map(lambda v: f"{v:.1f}" if not pd.isna(v) else "—")
if "trend_strength" in display_ft.columns:
    display_ft["trend_strength"] = display_ft["trend_strength"].map(
        lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
    )

st.dataframe(display_ft, use_container_width=True)
st.caption(f"Showing {len(ft):,} trading days of features.")

csv_bytes = ft.to_csv().encode("utf-8")
today_str = datetime.today().strftime("%Y-%m-%d")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name=f"{ticker_input}_features_{today_str}.csv",
    mime="text/csv",
)
```

- [ ] **Step 2: Run full test suite (UI has no unit tests)**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -q 2>&1 | tail -5
```

Expected: 130 passed.

- [ ] **Step 3: Commit and push**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add pages/4_Features.py && git commit -m "feat: Features page — signal snapshot, history, feature table export"
cd /Users/akhilm/Claude-Projects/influx-lab && git push origin main
```

---

## Done: Phase 4 Complete

At this point:
- ✅ 9 signal functions — volatility regime, momentum (short + long), mean reversion, trend strength, macro stress, volume regime, 52-week position, relative strength
- ✅ `feature_table.build()` — clean date × feature DataFrame for any ticker
- ✅ Signal Snapshot — radar chart (market fingerprint), colored badges, divergence flags
- ✅ Signal History — 2-year line chart + 1-year calendar heatmap
- ✅ Feature Table — downloadable CSV, ML-ready

**Deferred to Phase 5:**
- Regime state machine (multi-dimensional state labels)
- ML model training / backtesting
- "Days Like Today" pattern matching
- GenAI narrative layer
- Proprietary / niche signal development
