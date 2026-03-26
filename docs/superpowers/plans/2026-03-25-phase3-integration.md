# Phase 3: Integration (Market Relationship Lab) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Integration module — three analysis tools (cross-asset correlations, macro regime detection, lead-lag scanner) on top of the Phase 2 indicator set.

**Architecture:** Three focused modules under `modules/integration/` (`correlations.py`, `regimes.py`, `leadlag.py`), each with a single responsibility. `pages/3_Integration.py` replaces the stub. All data is sourced from existing Phase 2 indicator functions + the shared `fetch_prices` / `fetch_fred` data layer.

**Tech Stack:** Python 3.11+, Streamlit, pandas, numpy, scipy, plotly (graph_objects + express), pytest, unittest.mock

---

## File Map

```
modules/integration/
├── __init__.py            ← NEW: empty package marker
├── regimes.py             ← NEW: classify_regime, get_fear_greed_history, get_regime_history, get_regime_asset_stats
├── correlations.py        ← NEW: get_correlation_matrix
└── leadlag.py             ← NEW: cross_correlate, interpret_lag, scan_all_vs_target

tests/integration/
├── __init__.py            ← NEW: empty package marker
├── test_regimes.py        ← NEW
├── test_correlations.py   ← NEW
└── test_leadlag.py        ← NEW

pages/
└── 3_Integration.py       ← MODIFY: replace stub with full UI
```

**Key references (read before starting):**
- `modules/market_pulse/composite.py` — pattern for calling indicator functions + how `percentile_score` is used
- `modules/market_pulse/indicators.py` — all indicator functions; note `SECTOR_TICKERS`, `HEATMAP_TICKERS`, `HEATMAP_LABELS` constants at bottom
- `modules/market_pulse/scoring.py` — `percentile_score(series, value, invert)` returns `{score, color, label}`
- `modules/data/fetcher.py` — `fetch_prices(tickers, start, end)` and `fetch_fred(series_id, start, end)`
- `tests/market_pulse/test_composite.py` — example of how to mock indicator functions in tests

---

## Task 1: Module Scaffold

**Files:**
- Create: `modules/integration/__init__.py`
- Create: `tests/integration/__init__.py`

- [ ] **Step 1: Create both empty files**

```bash
touch /Users/akhilm/Claude-Projects/influx-lab/modules/integration/__init__.py
touch /Users/akhilm/Claude-Projects/influx-lab/tests/integration/__init__.py
```

- [ ] **Step 2: Verify pytest can discover the new package**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/ -v 2>&1 | head -10
```

Expected: "no tests ran" or empty collection — no errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/integration/__init__.py tests/integration/__init__.py && git commit -m "feat: scaffold integration module"
```

---

## Task 2: Regime Classification (`regimes.py`)

**Files:**
- Create: `modules/integration/regimes.py`
- Create: `tests/integration/test_regimes.py`

This module provides four functions:
- `classify_regime(score)` — maps a single 0–100 F&G score to a regime label
- `get_regime_history(fg_scores)` — maps a full series of scores to labels
- `get_fear_greed_history()` — reconstructs the daily F&G composite score over 3 years
- `get_regime_asset_stats(regime_history, asset_prices)` — per-regime mean log returns

### Step 1: Write failing tests

- [ ] **Step 1: Create `tests/integration/test_regimes.py`**

```python
import numpy as np
import pandas as pd
from unittest.mock import patch


# ─── classify_regime ──────────────────────────────────────────────────────────

def test_classify_regime_extreme_greed():
    from modules.integration.regimes import classify_regime
    assert classify_regime(0.0) == "Extreme Greed"
    assert classify_regime(10.0) == "Extreme Greed"
    assert classify_regime(19.9) == "Extreme Greed"


def test_classify_regime_greed():
    from modules.integration.regimes import classify_regime
    assert classify_regime(20.0) == "Greed"
    assert classify_regime(30.0) == "Greed"
    assert classify_regime(39.9) == "Greed"


def test_classify_regime_neutral():
    from modules.integration.regimes import classify_regime
    assert classify_regime(40.0) == "Neutral"
    assert classify_regime(50.0) == "Neutral"
    assert classify_regime(59.9) == "Neutral"


def test_classify_regime_fear():
    from modules.integration.regimes import classify_regime
    assert classify_regime(60.0) == "Fear"
    assert classify_regime(70.0) == "Fear"
    assert classify_regime(79.9) == "Fear"


def test_classify_regime_extreme_fear():
    from modules.integration.regimes import classify_regime
    assert classify_regime(80.0) == "Extreme Fear"
    assert classify_regime(100.0) == "Extreme Fear"


# ─── get_regime_history ────────────────────────────────────────────────────────

def test_get_regime_history_preserves_index():
    from modules.integration.regimes import get_regime_history
    idx = pd.date_range("2023-01-01", periods=5, freq="B")
    scores = pd.Series([10.0, 30.0, 50.0, 70.0, 90.0], index=idx)
    result = get_regime_history(scores)
    assert list(result.index) == list(idx)


def test_get_regime_history_correct_labels():
    from modules.integration.regimes import get_regime_history
    scores = pd.Series([10.0, 30.0, 50.0, 70.0, 90.0])
    result = get_regime_history(scores)
    assert list(result) == ["Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear"]


# ─── get_regime_asset_stats ────────────────────────────────────────────────────

def test_get_regime_asset_stats_returns_dataframe():
    from modules.integration.regimes import get_regime_asset_stats
    idx = pd.date_range("2023-01-01", periods=20, freq="B")
    regime_history = pd.Series(["Fear"] * 20, index=idx)
    prices = pd.DataFrame({"SPY": np.linspace(400, 420, 20)}, index=idx)
    result = get_regime_asset_stats(regime_history, prices)
    assert isinstance(result, pd.DataFrame)


def test_get_regime_asset_stats_has_all_five_regimes():
    from modules.integration.regimes import get_regime_asset_stats
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    labels = (["Extreme Greed"] * 20 + ["Greed"] * 20 +
              ["Neutral"] * 20 + ["Fear"] * 20 + ["Extreme Fear"] * 20)
    regime_history = pd.Series(labels, index=idx)
    prices = pd.DataFrame({"SPY": np.linspace(400, 500, 100)}, index=idx)
    result = get_regime_asset_stats(regime_history, prices)
    expected = {"Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear"}
    assert set(result.index) == expected


def test_get_regime_asset_stats_nan_for_sparse_regime():
    from modules.integration.regimes import get_regime_asset_stats
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    labels = ["Fear"] * 3 + ["Neutral"] * 7
    regime_history = pd.Series(labels, index=idx)
    prices = pd.DataFrame({"SPY": np.linspace(400, 410, 10)}, index=idx)
    result = get_regime_asset_stats(regime_history, prices)
    assert np.isnan(result.loc["Fear", "SPY"])


# ─── get_fear_greed_history ────────────────────────────────────────────────────

def _make_ind(idx, values, invert=False):
    return {
        "series": pd.Series(list(values), index=idx),
        "current": float(list(values)[-1]),
        "invert": invert,
        "label": "Test",
        "unit": "x",
    }


def test_get_fear_greed_history_returns_series():
    from modules.integration.regimes import get_fear_greed_history
    from unittest.mock import patch
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    ind = _make_ind(idx, range(100))
    targets = [
        "modules.market_pulse.indicators.get_vix",
        "modules.market_pulse.indicators.get_hy_credit_spread",
        "modules.market_pulse.indicators.get_dxy",
        "modules.market_pulse.indicators.get_yield_curve",
        "modules.market_pulse.indicators.get_gold_spy_ratio",
        "modules.market_pulse.indicators.get_aaii_sentiment",
    ]
    patches = [patch(t, return_value=ind) for t in targets]
    for p in patches:
        p.start()
    try:
        result = get_fear_greed_history()
    finally:
        for p in patches:
            p.stop()
    assert isinstance(result, pd.Series)
    assert len(result) > 0
    assert result.between(0, 100).all()


def test_get_fear_greed_history_excludes_sparse_dates():
    from modules.integration.regimes import get_fear_greed_history
    from unittest.mock import patch
    # Two indicators on different date ranges — overlap dates should be included, sparse excluded
    idx_a = pd.date_range("2023-01-01", periods=50, freq="B")
    idx_b = pd.date_range("2023-01-01", periods=10, freq="B")  # only 10 days
    ind_a = _make_ind(idx_a, range(50))
    ind_b = _make_ind(idx_b, range(10))
    # 5 indicators return ind_a, 1 returns ind_b — all 50 dates have ≥3 indicators
    with patch("modules.market_pulse.indicators.get_vix",              return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_hy_credit_spread", return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_dxy",              return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_yield_curve",      return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_gold_spy_ratio",   return_value=ind_a), \
         patch("modules.market_pulse.indicators.get_aaii_sentiment",   return_value=ind_b):
        result = get_fear_greed_history()
    # All 50 dates should be included (5 indicators cover all 50 dates)
    assert len(result) == 50
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/test_regimes.py -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError` — regimes.py doesn't exist yet.

- [ ] **Step 3: Create `modules/integration/regimes.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/test_regimes.py -v 2>&1 | tail -15
```

Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/integration/regimes.py tests/integration/test_regimes.py && git commit -m "feat: add regime classification module"
```

---

## Task 3: Correlation Matrix (`correlations.py`)

**Files:**
- Create: `modules/integration/correlations.py`
- Create: `tests/integration/test_correlations.py`

### Step 1: Write failing tests

- [ ] **Step 1: Create `tests/integration/test_correlations.py`**

```python
import numpy as np
import pandas as pd


def _make_price_df(n=90, seed=42):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(seed)
    return pd.DataFrame({
        "VIX": np.random.normal(20, 3, n),
        "DXY": np.random.normal(100, 2, n),
        "SPY": np.random.normal(450, 10, n),
    }, index=idx)


def test_get_correlation_matrix_is_square():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    assert result.shape[0] == result.shape[1] == 3


def test_get_correlation_matrix_diagonal_is_one():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    for col in result.columns:
        assert abs(result.loc[col, col] - 1.0) < 1e-9


def test_get_correlation_matrix_values_bounded():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    assert (result.fillna(0).values >= -1.0 - 1e-9).all()
    assert (result.fillna(0).values <= 1.0 + 1e-9).all()


def test_get_correlation_matrix_is_symmetric():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    result = get_correlation_matrix(prices)
    pd.testing.assert_frame_equal(result, result.T)


def test_get_correlation_matrix_window_uses_tail():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df(n=90)
    result_30 = get_correlation_matrix(prices, window=30)
    result_60 = get_correlation_matrix(prices, window=60)
    # Different windows → different matrices (with random data they won't match)
    assert not result_30.equals(result_60)


def test_get_correlation_matrix_regime_filter_uses_given_dates():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df(n=90)
    # Filter to first 30 dates
    regime_dates = prices.index[:30]
    result_regime = get_correlation_matrix(prices, regime_dates=regime_dates)
    result_first30 = get_correlation_matrix(prices.iloc[:30])
    pd.testing.assert_frame_equal(result_regime, result_first30)


def test_get_correlation_matrix_handles_nan():
    from modules.integration.correlations import get_correlation_matrix
    prices = _make_price_df()
    prices.iloc[10:20, 0] = np.nan
    result = get_correlation_matrix(prices)
    # Should not raise; diagonal still 1.0
    assert abs(result.loc["VIX", "VIX"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/test_correlations.py -v 2>&1 | tail -12
```

Expected: `ModuleNotFoundError` — correlations.py doesn't exist yet.

- [ ] **Step 3: Create `modules/integration/correlations.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/test_correlations.py -v 2>&1 | tail -12
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/integration/correlations.py tests/integration/test_correlations.py && git commit -m "feat: add correlation matrix module"
```

---

## Task 4: Lead-Lag Scanner (`leadlag.py`)

**Files:**
- Create: `modules/integration/leadlag.py`
- Create: `tests/integration/test_leadlag.py`

### Step 1: Write failing tests

- [ ] **Step 1: Create `tests/integration/test_leadlag.py`**

```python
import numpy as np
import pandas as pd


def _make_lagged_pair(n=300, lag=15, seed=42):
    """Two series where series_a leads series_b by `lag` trading days."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(seed)
    a_vals = np.random.normal(0, 1, n)
    b_vals = np.roll(a_vals, lag)
    b_vals[:lag] = 0.0
    a = pd.Series(a_vals, index=idx, name="a")
    b = pd.Series(b_vals + np.random.normal(0, 0.05, n), index=idx, name="b")
    return a, b


# ─── cross_correlate ──────────────────────────────────────────────────────────

def test_cross_correlate_returns_correct_index():
    from modules.integration.leadlag import cross_correlate
    a, b = _make_lagged_pair()
    result = cross_correlate(a, b, max_lag=30)
    assert list(result.index) == list(range(-30, 31))


def test_cross_correlate_lag0_matches_pearson():
    from modules.integration.leadlag import cross_correlate
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(99)
    a = pd.Series(np.random.normal(0, 1, 100), index=idx)
    b = pd.Series(np.random.normal(0, 1, 100), index=idx)
    result = cross_correlate(a, b, max_lag=10)
    expected = float(a.corr(b))
    assert abs(result[0] - expected) < 1e-9


def test_cross_correlate_detects_positive_lag():
    from modules.integration.leadlag import cross_correlate
    a, b = _make_lagged_pair(n=300, lag=15)
    result = cross_correlate(a, b, max_lag=30)
    peak_lag = int(result.abs().idxmax())
    assert 10 <= peak_lag <= 20


# ─── interpret_lag ────────────────────────────────────────────────────────────

def test_interpret_lag_positive():
    from modules.integration.leadlag import interpret_lag
    result = interpret_lag("Yield Curve", "VIX", peak_lag=45, peak_corr=0.71)
    assert "leads" in result
    assert "45" in result


def test_interpret_lag_negative():
    from modules.integration.leadlag import interpret_lag
    result = interpret_lag("XLK", "SPY", peak_lag=-3, peak_corr=0.88)
    assert "lags" in result
    assert "3" in result


def test_interpret_lag_zero():
    from modules.integration.leadlag import interpret_lag
    result = interpret_lag("VIX", "SPY", peak_lag=0, peak_corr=-0.75)
    assert "no lead" in result.lower() or "moves with" in result.lower()


# ─── scan_all_vs_target ───────────────────────────────────────────────────────

def test_scan_all_vs_target_returns_dataframe():
    from modules.integration.leadlag import scan_all_vs_target
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)
    target = pd.Series(np.random.normal(0, 1, 200), index=idx, name="VIX")
    indicators = {
        "DXY": pd.Series(np.random.normal(0, 1, 200), index=idx),
        "SPY": pd.Series(np.random.normal(0, 1, 200), index=idx),
    }
    result = scan_all_vs_target(target, indicators, max_lag=10)
    assert isinstance(result, pd.DataFrame)


def test_scan_all_vs_target_has_required_columns():
    from modules.integration.leadlag import scan_all_vs_target
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    np.random.seed(42)
    target = pd.Series(np.random.normal(0, 1, 200), index=idx, name="VIX")
    indicators = {"DXY": pd.Series(np.random.normal(0, 1, 200), index=idx)}
    result = scan_all_vs_target(target, indicators, max_lag=10)
    assert all(c in result.columns for c in ["indicator_name", "peak_correlation", "peak_lag", "interpretation"])


def test_scan_all_vs_target_sorted_by_abs_peak_correlation():
    from modules.integration.leadlag import scan_all_vs_target
    idx = pd.date_range("2023-01-01", periods=300, freq="B")
    np.random.seed(7)
    target = pd.Series(np.random.normal(0, 1, 300), index=idx, name="VIX")
    # strong: 90% correlated with target
    strong = target * 0.9 + pd.Series(np.random.normal(0, 0.1, 300), index=idx)
    weak = pd.Series(np.random.normal(0, 1, 300), index=idx)
    indicators = {"strong": strong, "weak": weak}
    result = scan_all_vs_target(target, indicators, max_lag=10)
    assert result.iloc[0]["indicator_name"] == "strong"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/test_leadlag.py -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError` — leadlag.py doesn't exist yet.

- [ ] **Step 3: Create `modules/integration/leadlag.py`**

```python
import pandas as pd
import numpy as np


def cross_correlate(series_a: pd.Series, series_b: pd.Series, max_lag: int = 60) -> pd.Series:
    """
    Pearson correlation of series_a[t] vs series_b[t+lag] for lag in [-max_lag, +max_lag].

    Positive lag: series_a LEADS series_b (a moves first, b follows).
    Negative lag: series_a LAGS series_b (b moves first, a follows).

    Series are aligned on common dates before computation.
    Returns Series indexed by integer lag.
    """
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    a, b = aligned["a"], aligned["b"]

    results = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = b.shift(-lag)
        pair = pd.concat([a, shifted], axis=1).dropna()
        results[lag] = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 1 else float("nan")
    return pd.Series(results)


def interpret_lag(indicator_name: str, target_name: str, peak_lag: int, peak_corr: float) -> str:
    """Plain-language interpretation of a cross-correlation result."""
    if peak_lag > 0:
        return f"{indicator_name} leads {target_name} by ~{peak_lag} days (corr: {peak_corr:.2f})"
    elif peak_lag < 0:
        return f"{indicator_name} lags {target_name} by ~{abs(peak_lag)} days (corr: {peak_corr:.2f})"
    else:
        return f"{indicator_name} moves with {target_name} (no lead/lag, corr: {peak_corr:.2f})"


def scan_all_vs_target(
    target: pd.Series,
    indicators: dict[str, pd.Series],
    max_lag: int = 60,
) -> pd.DataFrame:
    """
    Scan all indicators vs target using cross_correlate.
    Returns DataFrame sorted by abs(peak_correlation) descending.
    Columns: indicator_name, peak_correlation, peak_lag, interpretation.
    """
    target_name = target.name or "target"
    rows = []
    for name, series in indicators.items():
        cc = cross_correlate(series, target, max_lag)
        cc_abs = cc.abs()
        peak_lag = int(cc_abs.idxmax())
        peak_corr = float(cc.loc[peak_lag])
        rows.append({
            "indicator_name": name,
            "peak_correlation": round(peak_corr, 4),
            "peak_lag": peak_lag,
            "interpretation": interpret_lag(name, target_name, peak_lag, peak_corr),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("peak_correlation", key=lambda s: s.abs(), ascending=False)
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run all integration tests**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/integration/ -v 2>&1 | tail -20
```

Expected: All tests PASS (12 regimes + 7 correlations + 9 leadlag = 28 total).

- [ ] **Step 5: Run full suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -q 2>&1 | tail -5
```

Expected: All tests PASS (84 existing + 28 new = 112 total).

- [ ] **Step 6: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/integration/leadlag.py tests/integration/test_leadlag.py && git commit -m "feat: add lead-lag scanner module"
```

---

## Task 5: Integration UI (`pages/3_Integration.py`)

**Files:**
- Modify: `pages/3_Integration.py`

No unit tests — UI layer only. Visual verification in browser.

**Before writing:** Read `pages/2_Market_Pulse.py` to understand the layout patterns (section headers, sparkline helpers, error handling pattern).

- [ ] **Step 1: Replace `pages/3_Integration.py`**

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from modules.integration.correlations import get_correlation_matrix
from modules.integration.regimes import (
    get_fear_greed_history, get_regime_history, get_regime_asset_stats,
    REGIME_ORDER, REGIME_COLORS,
)
from modules.integration.leadlag import scan_all_vs_target, cross_correlate
from modules.market_pulse.indicators import (
    get_vix, get_dxy, get_hy_credit_spread,
    get_yield_curve, get_gold_spy_ratio, get_copper_gold_ratio,
    SECTOR_TICKERS, HEATMAP_TICKERS, HEATMAP_LABELS,
)
from modules.data.fetcher import fetch_prices

st.set_page_config(page_title="Integration", layout="wide")
st.title("Integration")
st.caption("Cross-asset relationships, macro regime history, and lead-lag analysis. A hypothesis generator, not a prediction engine.")


# ─── Data Loading ──────────────────────────────────────────────────────────────

def _load_daily_series() -> dict[str, pd.Series]:
    """Load all daily-frequency series for this page."""
    series = {}
    for fn, name in [
        (get_vix,              "VIX"),
        (get_dxy,              "DXY"),
        (get_hy_credit_spread, "HY Spread"),
        (get_yield_curve,      "Yield Curve"),
        (get_gold_spy_ratio,   "Gold/SPY"),
        (get_copper_gold_ratio,"Copper/Gold"),
    ]:
        try:
            series[name] = fn()["series"]
        except Exception:
            pass

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=1100)).strftime("%Y-%m-%d")
    extra_tickers = list(SECTOR_TICKERS.keys()) + list(HEATMAP_TICKERS)
    try:
        prices = fetch_prices(extra_tickers, start, end)
        for ticker in extra_tickers:
            if ticker in prices.columns:
                label = SECTOR_TICKERS.get(ticker) or HEATMAP_LABELS.get(ticker, ticker)
                series[label] = prices[ticker].dropna()
    except Exception:
        pass

    return series


all_series = _load_daily_series()
all_prices_df = pd.DataFrame(all_series) if all_series else pd.DataFrame()


# ─── Section 1: Cross-Asset Correlations ──────────────────────────────────────
st.header("Cross-Asset Correlations")
st.caption("How do these assets move relative to each other? Green = move together, red = move opposite.")

try:
    if all_prices_df.empty:
        st.warning("No asset data available.")
    else:
        col_win, col_reg = st.columns([1, 2])
        with col_win:
            window = st.selectbox("Window", [30, 60, 90], index=1, key="corr_window")
        with col_reg:
            regime_filter = st.selectbox(
                "Regime filter",
                ["All periods"] + REGIME_ORDER,
                index=0,
                key="corr_regime",
            )

        regime_dates = None
        if regime_filter != "All periods":
            try:
                fg_hist = get_fear_greed_history()
                r_hist  = get_regime_history(fg_hist)
                regime_dates = r_hist[r_hist == regime_filter].index
            except Exception:
                st.warning("Could not load regime history for filtering.")

        corr = get_correlation_matrix(all_prices_df, window=window, regime_dates=regime_dates)

        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_corr.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_colorbar=dict(title="Correlation"),
        )
        fig_corr.update_traces(textfont_size=8)
        st.plotly_chart(fig_corr, use_container_width=True)

        if regime_filter != "All periods":
            n_days = len(regime_dates) if regime_dates is not None else 0
            st.caption(f"Showing correlations during **{regime_filter}** periods only ({n_days} trading days).")
        else:
            st.caption(
                "In a crisis, assets that normally move independently tend to move together "
                "— that's when diversification fails. Try filtering by 'Extreme Fear' to see this."
            )

except Exception as e:
    st.error(f"Could not load correlations: {e}")

st.divider()

# ─── Section 2: Macro Regime ──────────────────────────────────────────────────
st.header("Macro Regime")
st.caption("Market environment classified from the Fear & Greed composite score over 3 years.")

REGIME_DESCRIPTIONS = {
    "Extreme Greed": "Markets are very calm. Risk appetite is elevated, volatility low, credit conditions easy.",
    "Greed":         "Conditions are favorable. Investors are taking on risk, spreads are tight.",
    "Neutral":       "Mixed signals. No dominant fear or greed — watch for a shift in either direction.",
    "Fear":          "Stress is building. Credit spreads widening, volatility rising, defensive rotation starting.",
    "Extreme Fear":  "Fear is dominant. Investors fleeing to safety — gold, bonds, cash.",
}

try:
    fg_history     = get_fear_greed_history()
    regime_history = get_regime_history(fg_history)
    current_regime = regime_history.iloc[-1]
    current_color  = REGIME_COLORS[current_regime]

    st.markdown(
        f"<div style='padding:16px;background:{current_color}20;"
        f"border-left:4px solid {current_color};border-radius:4px'>"
        f"<b style='font-size:1.1em'>Current Regime: "
        f"<span style='color:{current_color}'>{current_regime}</span></b><br>"
        f"{REGIME_DESCRIPTIONS.get(current_regime, '')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Regime timeline
    fig_tl = go.Figure()
    for regime in REGIME_ORDER:
        mask  = regime_history == regime
        dates = fg_history[mask].index
        scores = fg_history[mask].values
        fig_tl.add_trace(go.Scatter(
            x=dates, y=scores,
            mode="markers",
            marker=dict(color=REGIME_COLORS[regime], size=4),
            name=regime,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} (" + regime + ")<extra></extra>",
        ))
    for y0, y1 in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        regime_for_band = REGIME_ORDER[int(y0 / 20)]
        fig_tl.add_hrect(y0=y0, y1=y1, fillcolor=REGIME_COLORS[regime_for_band], opacity=0.05, line_width=0)
    fig_tl.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="F&G Score", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # Per-regime asset stats
    st.markdown("**Average daily return by regime** (key indicators)")
    STAT_ASSETS = ["VIX", "DXY", "HY Spread", "Yield Curve", "Gold/SPY", "Copper/Gold"]
    stat_df_input = all_prices_df[[c for c in STAT_ASSETS if c in all_prices_df.columns]]
    if not stat_df_input.empty:
        stats = get_regime_asset_stats(regime_history, stat_df_input)
        fmt = stats.apply(
            lambda col: col.map(lambda v: f"{v:.3%}" if not pd.isna(v) else "—")
        )
        st.dataframe(fmt, use_container_width=True)
    else:
        st.caption("No asset data available for regime stats.")

except Exception as e:
    st.error(f"Could not load regime data: {e}")

st.divider()

# ─── Section 3: Lead-Lag Scanner ──────────────────────────────────────────────
st.header("Lead-Lag Scanner")
st.caption(
    "Which indicators move before others? "
    "Pick a target, click Scan — we rank all indicators by their leading/lagging relationship."
)

TARGET_OPTIONS = ["VIX", "DXY", "HY Spread", "Yield Curve", "Gold/SPY", "Copper/Gold", "S&P 500 (US)"]

try:
    available_targets = [t for t in TARGET_OPTIONS if t in all_series]
    if not available_targets:
        st.warning("No target series available.")
    else:
        col_t, col_btn = st.columns([3, 1])
        with col_t:
            selected_target = st.selectbox("Target", available_targets, key="ll_target")
        with col_btn:
            st.write("")
            run_scan = st.button("Scan", key="ll_scan", type="primary")

        if run_scan:
            target_s = all_series[selected_target].rename(selected_target)
            scan_pool = {k: v for k, v in all_series.items() if k != selected_target}
            with st.spinner(f"Scanning {len(scan_pool)} indicators vs {selected_target}..."):
                results = scan_all_vs_target(target_s, scan_pool, max_lag=60)
            st.session_state["ll_results"]      = results
            st.session_state["ll_target_name"]  = selected_target

        if "ll_results" in st.session_state:
            results      = st.session_state["ll_results"]
            target_name  = st.session_state["ll_target_name"]

            if not results.empty:
                st.markdown(f"**Ranked relationships with {target_name}**")
                display = results[["indicator_name", "peak_correlation", "peak_lag", "interpretation"]].copy()
                display["peak_correlation"] = display["peak_correlation"].map(lambda v: f"{v:+.3f}")
                st.dataframe(display, use_container_width=True, hide_index=True)

                st.markdown("**Lag profile for selected indicator**")
                detail = st.selectbox(
                    "Inspect indicator",
                    results["indicator_name"].tolist(),
                    key="ll_detail",
                )
                if detail and detail in all_series:
                    cc = cross_correlate(
                        all_series[detail],
                        all_series[target_name].rename(target_name),
                        max_lag=60,
                    )
                    peak_lag = int(cc.abs().idxmax())

                    fig_cc = go.Figure(go.Bar(
                        x=cc.index.tolist(),
                        y=cc.values,
                        marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in cc.values],
                    ))
                    fig_cc.add_vline(x=0,        line_dash="dash", line_color="gray",   line_width=1)
                    fig_cc.add_vline(x=peak_lag, line_dash="dot",  line_color="#FFA500", line_width=2,
                                     annotation_text=f"peak: lag {peak_lag}",
                                     annotation_position="top right")
                    fig_cc.update_layout(
                        height=300, margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Lag (trading days)",
                        yaxis=dict(title="Correlation", range=[-1, 1]),
                    )
                    st.plotly_chart(fig_cc, use_container_width=True)
                    st.caption(
                        "Bars to the **right** (positive lag) = indicator moves before the target. "
                        "Bars to the **left** (negative lag) = indicator follows the target. "
                        "Orange line = peak correlation."
                    )

except Exception as e:
    st.error(f"Could not load lead-lag scanner: {e}")
```

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -q 2>&1 | tail -5
```

Expected: All 109 tests PASS.

- [ ] **Step 3: Commit and push**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add pages/3_Integration.py && git commit -m "feat: Integration page — correlations, regime timeline, lead-lag scanner"
cd /Users/akhilm/Claude-Projects/influx-lab && git push origin main
```

---

## Done: Phase 3 Complete

At this point:
- ✅ Cross-asset correlations — rolling heatmap with regime filter
- ✅ Macro regime timeline — 3-year F&G history, color-coded, per-regime asset stats
- ✅ Lead-lag scanner — scan all indicators vs any target, ranked results + lag profile chart

**Deferred to Phase 4:** Signal generation from discovered relationships, feature engineering, quantified hypothesis testing.
