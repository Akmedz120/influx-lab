# Phase 2b: Market Pulse Advanced — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Market Pulse module with a custom Fear & Greed composite index (gauge dial), VIX term structure, liquidity indicators (M2, Fed balance sheet), advanced international (FX pairs, Copper/Gold ratio), and sector rotation pulse.

**Architecture:** New `composite.py` for the Fear & Greed index computation. New indicator functions appended to existing `indicators.py`. The Market Pulse UI page gets new sections. Market breadth (500-ticker fetch) and COT/Insider data are deferred to a later phase due to complexity.

**Tech Stack:** Python 3.11+, Streamlit, yfinance, fredapi, pandas, numpy, scipy, plotly, pytest

---

## File Map

```
modules/market_pulse/
├── composite.py            ← NEW: Fear & Greed composite index
├── indicators.py           ← MODIFY: add 6 new indicator functions + constants
└── scoring.py              ← unchanged

tests/market_pulse/
├── test_composite.py       ← NEW: composite tests
└── test_indicators.py      ← MODIFY: append new indicator tests

pages/
└── 2_Market_Pulse.py       ← MODIFY: add Fear & Greed gauge + 4 new sections
```

**New indicator functions in `indicators.py`:**
- `get_vix_term_structure()` — VIX9D / VIX / VIX3M contango vs backwardation
- `get_m2()` — M2 money supply (FRED: M2SL)
- `get_fed_balance_sheet()` — Fed balance sheet (FRED: WALCL)
- `FX_TICKERS` + `get_fx_pairs()` — EUR/USD, USD/JPY, USD/CNY returns
- `get_copper_gold_ratio()` — growth vs fear signal
- `SECTOR_TICKERS` + `get_sector_performance()` — SPDR sector ETF performance

---

## Task 1: Fear & Greed Composite Index

**Files:**
- Create: `modules/market_pulse/composite.py`
- Create: `tests/market_pulse/test_composite.py`

- [ ] **Step 1: Write failing tests**

Create `tests/market_pulse/test_composite.py`:

```python
import pytest
import pandas as pd
from contextlib import ExitStack
from unittest.mock import patch


PATCH_TARGETS = [
    "modules.market_pulse.composite.get_vix",
    "modules.market_pulse.composite.get_hy_credit_spread",
    "modules.market_pulse.composite.get_dxy",
    "modules.market_pulse.composite.get_yield_curve",
    "modules.market_pulse.composite.get_gold_spy_ratio",
    "modules.market_pulse.composite.get_aaii_sentiment",
]


def make_indicator(current, series_values, invert=False):
    return {
        "series": pd.Series(list(series_values)),
        "current": float(current),
        "label": "Test",
        "unit": "x",
        "invert": invert,
    }


def mock_all(return_value):
    """Patch all 6 composite indicator functions with the same return value."""
    ctx = ExitStack()
    for target in PATCH_TARGETS:
        ctx.enter_context(patch(target, return_value=return_value))
    return ctx


def test_returns_required_keys():
    from modules.market_pulse.composite import compute_fear_greed_index
    ind = make_indicator(50, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert all(k in result for k in ["score", "label", "details"])


def test_score_is_float_between_0_and_100():
    from modules.market_pulse.composite import compute_fear_greed_index
    ind = make_indicator(50, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 100.0


def test_all_stressed_gives_high_score():
    from modules.market_pulse.composite import compute_fear_greed_index
    # current=99 against range(101) → ~99th pct → red
    ind = make_indicator(99, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert result["score"] > 75
    assert result["label"] in ["Fear", "Extreme Fear"]


def test_all_calm_gives_low_score():
    from modules.market_pulse.composite import compute_fear_greed_index
    # current=1 against range(101) → ~1st pct → green
    ind = make_indicator(1, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert result["score"] < 25
    assert result["label"] in ["Greed", "Extreme Greed"]


def test_details_has_one_entry_per_indicator():
    from modules.market_pulse.composite import compute_fear_greed_index
    ind = make_indicator(50, range(101))
    with mock_all(ind):
        result = compute_fear_greed_index()
    assert len(result["details"]) == 6


def test_failed_indicator_is_skipped_gracefully():
    from modules.market_pulse.composite import compute_fear_greed_index
    # 5 good, 1 raises — should still return a score from the 5
    ind = make_indicator(50, range(101))
    ctx = ExitStack()
    for i, target in enumerate(PATCH_TARGETS):
        if i == 0:
            ctx.enter_context(patch(target, side_effect=Exception("no data")))
        else:
            ctx.enter_context(patch(target, return_value=ind))
    with ctx:
        result = compute_fear_greed_index()
    assert len(result["details"]) == 5
    assert 0.0 <= result["score"] <= 100.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_composite.py -v
```

Expected: `ModuleNotFoundError` — composite.py doesn't exist yet.

- [ ] **Step 3: Implement composite.py**

Create `modules/market_pulse/composite.py`:

```python
from modules.market_pulse.indicators import (
    get_vix, get_hy_credit_spread, get_dxy,
    get_yield_curve, get_gold_spy_ratio, get_aaii_sentiment,
)
from modules.market_pulse.scoring import percentile_score

# 6 indicators used in composite. Order matters for display.
COMPOSITE_FUNCTIONS = [
    get_vix, get_hy_credit_spread, get_dxy,
    get_yield_curve, get_gold_spy_ratio, get_aaii_sentiment,
]


def compute_fear_greed_index() -> dict:
    """
    Custom Fear & Greed Index built from 6 Phase 2a indicators.

    Each indicator is scored 0–100 via 3-year percentile rank.
    Equal-weighted average gives the composite score.

    Score interpretation:
        0–20   : Extreme Greed (market very calm)
        20–40  : Greed
        40–60  : Neutral
        60–80  : Fear
        80–100 : Extreme Fear (market very stressed)

    Failed/unavailable indicators are skipped; remaining are averaged.
    Returns {"score": float, "label": str, "details": list[dict]}
    """
    scores = []
    details = []
    for fn in COMPOSITE_FUNCTIONS:
        try:
            ind = fn()
            sc = percentile_score(ind["series"], ind["current"], invert=ind.get("invert", False))
            scores.append(sc["score"])
            details.append({
                "label": ind["label"],
                "score": sc["score"],
                "color": sc["color"],
            })
        except Exception:
            pass

    if not scores:
        return {"score": 50.0, "label": "Neutral", "details": []}

    avg = round(sum(scores) / len(scores), 1)

    if avg >= 80:
        label = "Extreme Fear"
    elif avg >= 60:
        label = "Fear"
    elif avg >= 40:
        label = "Neutral"
    elif avg >= 20:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {"score": avg, "label": label, "details": details}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_composite.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/composite.py tests/market_pulse/test_composite.py && git commit -m "feat: add Fear & Greed composite index"
```

---

## Task 2: VIX Term Structure

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── VIX Term Structure ───────────────────────────────────────────────────────

def test_get_vix_term_structure_returns_dict_keys():
    from modules.market_pulse.indicators import get_vix_term_structure
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    mock_df = pd.DataFrame({
        "^VIX9D": [14.0] * 10,
        "^VIX":   [16.0] * 10,
        "^VIX3M": [18.0] * 10,
    }, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix_term_structure()
    assert all(k in result for k in ["vix9d", "vix", "vix3m", "structure", "label"])


def test_get_vix_term_structure_contango_when_vix9d_below_vix3m():
    from modules.market_pulse.indicators import get_vix_term_structure
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    mock_df = pd.DataFrame({
        "^VIX9D": [12.0] * 5,
        "^VIX":   [15.0] * 5,
        "^VIX3M": [18.0] * 5,
    }, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix_term_structure()
    assert "Contango" in result["structure"]


def test_get_vix_term_structure_backwardation_when_vix9d_above_vix3m():
    from modules.market_pulse.indicators import get_vix_term_structure
    idx = pd.date_range("2025-01-01", periods=5, freq="B")
    mock_df = pd.DataFrame({
        "^VIX9D": [30.0] * 5,
        "^VIX":   [25.0] * 5,
        "^VIX3M": [20.0] * 5,
    }, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix_term_structure()
    assert "Backwardation" in result["structure"]
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_vix_term_structure_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_vix_term_structure_contango_when_vix9d_below_vix3m tests/market_pulse/test_indicators.py::test_get_vix_term_structure_backwardation_when_vix9d_above_vix3m -v
```

Expected: `ImportError` — function not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
def get_vix_term_structure() -> dict:
    """
    Fetch VIX9D (^VIX9D), VIX (^VIX), VIX3M (^VIX3M) from yfinance.

    Contango:      VIX9D < VIX < VIX3M  — normal, market calm
    Backwardation: VIX9D > VIX > VIX3M  — panic, near-term fear elevated

    Returns current values for all three + structure label.
    Note: VIX3M was formerly ^VXV; yfinance may serve either ticker.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["^VIX9D", "^VIX", "^VIX3M"], start, end)

    def _last(col):
        if col not in prices.columns:
            return None
        s = prices[col].dropna()
        return float(s.iloc[-1]) if len(s) > 0 else None

    vix9d = _last("^VIX9D")
    vix   = _last("^VIX")
    vix3m = _last("^VIX3M")

    if vix9d is not None and vix3m is not None:
        structure = "Backwardation (Panic)" if vix9d > vix3m else "Contango (Calm)"
    else:
        structure = "Unknown"

    return {
        "vix9d": vix9d,
        "vix": vix,
        "vix3m": vix3m,
        "structure": structure,
        "label": "VIX Term Structure",
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_vix_term_structure_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_vix_term_structure_contango_when_vix9d_below_vix3m tests/market_pulse/test_indicators.py::test_get_vix_term_structure_backwardation_when_vix9d_above_vix3m -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add VIX term structure indicator"
```

---

## Task 3: Liquidity Indicators (M2, Fed Balance Sheet)

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── Liquidity ────────────────────────────────────────────────────────────────

def test_get_m2_returns_dict_keys():
    from modules.market_pulse.indicators import get_m2
    monthly = pd.Series(
        [20000.0 + i * 10 for i in range(36)],
        index=pd.date_range("2023-01-01", periods=36, freq="ME")
    )
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=monthly):
        result = get_m2()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_m2_invert_is_true():
    from modules.market_pulse.indicators import get_m2
    s = pd.Series([20000.0] * 36, index=pd.date_range("2023-01-01", periods=36, freq="ME"))
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=s):
        result = get_m2()
    assert result["invert"] is True


def test_get_fed_balance_sheet_returns_dict_keys():
    from modules.market_pulse.indicators import get_fed_balance_sheet
    weekly = pd.Series(
        [8_000_000.0 + i * 1000 for i in range(150)],
        index=pd.date_range("2023-01-01", periods=150, freq="W")
    )
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=weekly):
        result = get_fed_balance_sheet()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_fed_balance_sheet_current_is_last():
    from modules.market_pulse.indicators import get_fed_balance_sheet
    values = [8_000_000.0 + i * 1000 for i in range(50)]
    s = pd.Series(values, index=pd.date_range("2023-01-01", periods=50, freq="W"))
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=s):
        result = get_fed_balance_sheet()
    assert abs(result["current"] - values[-1]) < 1.0
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_m2_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_m2_invert_is_true tests/market_pulse/test_indicators.py::test_get_fed_balance_sheet_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_fed_balance_sheet_current_is_last -v
```

Expected: `ImportError` — functions not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
def get_m2() -> dict:
    """
    Fetch M2 money supply (FRED: M2SL). Monthly series, units in billions USD.
    High/rising M2 = more liquidity = calm market conditions.
    Score with invert=True.
    """
    start, end = _start(), _today()
    series = fetch_fred("M2SL", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "M2 Money Supply",
        "unit": "$B",
        "invert": True,
    }


def get_fed_balance_sheet() -> dict:
    """
    Fetch Fed balance sheet total assets (FRED: WALCL). Weekly series, units in millions USD.
    Shrinking balance sheet = tighter liquidity = stressed.
    Score with invert=True (large balance sheet = more liquidity = calm).
    """
    start, end = _start(), _today()
    series = fetch_fred("WALCL", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Fed Balance Sheet",
        "unit": "$M",
        "invert": True,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_m2_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_m2_invert_is_true tests/market_pulse/test_indicators.py::test_get_fed_balance_sheet_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_fed_balance_sheet_current_is_last -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add M2 and Fed balance sheet liquidity indicators"
```

---

## Task 4: Advanced International (FX pairs, Copper/Gold)

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── Advanced International ───────────────────────────────────────────────────

def test_get_fx_pairs_returns_dataframe():
    from modules.market_pulse.indicators import get_fx_pairs, FX_TICKERS
    idx = pd.date_range("2025-01-01", periods=14, freq="B")
    mock_df = pd.DataFrame(
        {t: np.linspace(1.0, 1.05, 14) for t in FX_TICKERS},
        index=idx
    )
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_fx_pairs()
    assert isinstance(result, pd.DataFrame)


def test_get_fx_pairs_has_required_columns():
    from modules.market_pulse.indicators import get_fx_pairs, FX_TICKERS
    idx = pd.date_range("2025-01-01", periods=14, freq="B")
    mock_df = pd.DataFrame(
        {t: np.linspace(1.0, 1.05, 14) for t in FX_TICKERS},
        index=idx
    )
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_fx_pairs()
    assert all(c in result.columns for c in ["ticker", "label", "current", "1d", "1w"])


def test_get_copper_gold_ratio_returns_dict_keys():
    from modules.market_pulse.indicators import get_copper_gold_ratio
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({
        "HG=F": np.full(500, 4.0),
        "GC=F": np.full(500, 2000.0),
    }, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_copper_gold_ratio()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_copper_gold_ratio_value_is_copper_over_gold():
    from modules.market_pulse.indicators import get_copper_gold_ratio
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_df = pd.DataFrame({
        "HG=F": np.full(100, 4.0),
        "GC=F": np.full(100, 2000.0),
    }, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_copper_gold_ratio()
    assert abs(result["current"] - (4.0 / 2000.0)) < 1e-6
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_fx_pairs_returns_dataframe tests/market_pulse/test_indicators.py::test_get_fx_pairs_has_required_columns tests/market_pulse/test_indicators.py::test_get_copper_gold_ratio_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_copper_gold_ratio_value_is_copper_over_gold -v
```

Expected: `ImportError` — not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
FX_TICKERS = {
    "EURUSD=X": "EUR / USD",
    "JPY=X":    "USD / JPY",
    "CNY=X":    "USD / CNY",
}


def get_fx_pairs() -> pd.DataFrame:
    """
    Fetch EUR/USD, USD/JPY, USD/CNY from yfinance over a 14-day window.
    Returns DataFrame: ticker, label, current, 1d_return, 1w_return.
    """
    end = _today()
    start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
    tickers = list(FX_TICKERS.keys())
    prices = fetch_prices(tickers, start, end)

    rows = []
    for ticker, label in FX_TICKERS.items():
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        current = float(s.iloc[-1])
        r1d = float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) >= 2 else None
        r1w = float(s.iloc[-1] / s.iloc[-6] - 1) if len(s) >= 6 else None
        rows.append({"ticker": ticker, "label": label, "current": current, "1d": r1d, "1w": r1w})
    return pd.DataFrame(rows)


def get_copper_gold_ratio() -> dict:
    """
    Compute Copper (HG=F) / Gold (GC=F) ratio.
    High ratio = economic optimism (growth > fear) = calm.
    Low ratio = flight to safety = stressed.
    Score with invert=True.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["HG=F", "GC=F"], start, end)
    ratio = (prices["HG=F"] / prices["GC=F"]).dropna()
    return {
        "series": ratio,
        "current": float(ratio.iloc[-1]),
        "label": "Copper / Gold Ratio",
        "unit": "ratio",
        "invert": True,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_fx_pairs_returns_dataframe tests/market_pulse/test_indicators.py::test_get_fx_pairs_has_required_columns tests/market_pulse/test_indicators.py::test_get_copper_gold_ratio_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_copper_gold_ratio_value_is_copper_over_gold -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add FX pairs and Copper/Gold ratio indicators"
```

---

## Task 5: Sector Rotation Pulse

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── Sector Rotation ──────────────────────────────────────────────────────────

def test_get_sector_performance_returns_dataframe():
    from modules.market_pulse.indicators import get_sector_performance, SECTOR_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame(
        {t: np.linspace(100, 110, 30) for t in SECTOR_TICKERS},
        index=idx
    )
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_sector_performance()
    assert isinstance(result, pd.DataFrame)


def test_get_sector_performance_has_required_columns():
    from modules.market_pulse.indicators import get_sector_performance, SECTOR_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame(
        {t: np.linspace(100, 110, 30) for t in SECTOR_TICKERS},
        index=idx
    )
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_sector_performance()
    assert all(c in result.columns for c in ["ticker", "label", "1d", "1w", "1m", "risk_type"])


def test_get_sector_performance_risk_type_values():
    from modules.market_pulse.indicators import get_sector_performance, SECTOR_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame(
        {t: np.linspace(100, 110, 30) for t in SECTOR_TICKERS},
        index=idx
    )
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_sector_performance()
    assert set(result["risk_type"].unique()).issubset({"Risk-On", "Risk-Off"})
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_sector_performance_returns_dataframe tests/market_pulse/test_indicators.py::test_get_sector_performance_has_required_columns tests/market_pulse/test_indicators.py::test_get_sector_performance_risk_type_values -v
```

Expected: `ImportError` — not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
SECTOR_TICKERS = {
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

_RISK_ON_SECTORS = {"XLK", "XLY", "XLF", "XLI", "XLE"}


def get_sector_performance() -> pd.DataFrame:
    """
    Fetch SPDR sector ETF 1d/1w/1m performance.
    Labels each sector as Risk-On or Risk-Off.

    Risk-On:  XLK, XLY, XLF, XLI, XLE  (grow with economy)
    Risk-Off: XLP, XLU, XLRE, XLV, XLB (defensive)

    Returns DataFrame: ticker, label, 1d, 1w, 1m, risk_type.
    """
    end = _today()
    start = (datetime.today() - timedelta(days=40)).strftime("%Y-%m-%d")
    tickers = list(SECTOR_TICKERS.keys())
    prices = fetch_prices(tickers, start, end)

    rows = []
    for ticker, label in SECTOR_TICKERS.items():
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        r1d = float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) >= 2 else None
        r1w = float(s.iloc[-1] / s.iloc[-6] - 1) if len(s) >= 6 else None
        r1m = float(s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 20 else None
        rows.append({
            "ticker": ticker,
            "label": label,
            "1d": r1d,
            "1w": r1w,
            "1m": r1m,
            "risk_type": "Risk-On" if ticker in _RISK_ON_SECTORS else "Risk-Off",
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run all indicator + composite tests**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/ -v
```

Expected: All tests PASS (should be 25 indicator + 6 composite + 3 new VIX + 4 liquidity + 4 FX/copper + 3 sector = ~45 total in market_pulse).

- [ ] **Step 5: Run full suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add sector rotation pulse indicator"
```

---

## Task 6: Update Market Pulse UI

**Files:**
- Modify: `pages/2_Market_Pulse.py`

No unit tests — UI layer only.

- [ ] **Step 1: Replace pages/2_Market_Pulse.py**

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.market_pulse.composite import compute_fear_greed_index
from modules.market_pulse.indicators import (
    get_yield_curve, get_fed_funds, get_vix,
    get_aaii_sentiment, get_consumer_confidence,
    get_hy_credit_spread, get_gold_spy_ratio, get_dxy,
    get_global_heatmap, get_vix_term_structure,
    get_m2, get_fed_balance_sheet,
    get_fx_pairs, get_copper_gold_ratio,
    get_sector_performance,
)
from modules.market_pulse.scoring import percentile_score

st.set_page_config(page_title="Market Pulse", layout="wide")
st.title("Market Pulse")
st.caption("Daily read on market mood and stress — color-coded green/yellow/red against 3-year history.")

COLOR_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
COLOR_HEX   = {"green": "#00CC96", "yellow": "#FFA500", "red": "#EF553B"}
FEAR_COLORS = {
    "Extreme Greed": "#00CC96",
    "Greed":         "#7FBA00",
    "Neutral":       "#FFA500",
    "Fear":          "#FF6B35",
    "Extreme Fear":  "#EF553B",
}


def _score(ind: dict) -> dict:
    return percentile_score(ind["series"], ind["current"], invert=ind.get("invert", False))


def _sparkline(ind: dict, sc: dict, height: int = 150):
    fig = go.Figure(go.Scatter(
        y=ind["series"].values,
        x=ind["series"].index,
        mode="lines",
        line=dict(color=COLOR_HEX[sc["color"]], width=1.5),
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=True, tickformat=".2f"),
    )
    return fig


def _indicator_block(col, ind: dict, sc: dict, extra_note: str | None = None):
    with col:
        st.metric(
            label=f"{COLOR_EMOJI[sc['color']]} {ind['label']}",
            value=f"{ind['current']:.2f} {ind['unit']}",
        )
        st.caption(f"{sc['label']} · {sc['score']:.0f}th pct")
        if extra_note:
            st.info(extra_note)
        st.plotly_chart(_sparkline(ind, sc), use_container_width=True)


def _color_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    color = "#00CC96" if val >= 0 else "#EF553B"
    return f'<span style="color:{color}">{val:.2%}</span>'


# ─── Fear & Greed Gauge ───────────────────────────────────────────────────────
try:
    fg = compute_fear_greed_index()
    gauge_color = FEAR_COLORS.get(fg["label"], "#FFA500")

    col_gauge, col_detail = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fg["score"],
            number={"font": {"size": 48}},
            title={"text": f"<b>Fear & Greed</b><br><span style='font-size:1.2em;color:{gauge_color}'>{fg['label']}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 20],   "color": "#00CC96"},
                    {"range": [20, 40],  "color": "#7FBA00"},
                    {"range": [40, 60],  "color": "#FFA500"},
                    {"range": [60, 80],  "color": "#FF6B35"},
                    {"range": [80, 100], "color": "#EF553B"},
                ],
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_detail:
        st.markdown("**Component Breakdown**")
        for d in fg["details"]:
            emoji = COLOR_EMOJI.get(d["color"], "⚪")
            st.markdown(f"{emoji} **{d['label']}** — {d['score']:.0f}th pct")

except Exception as e:
    st.error(f"Fear & Greed index unavailable: {e}")

st.divider()

# ─── Summary Strip ────────────────────────────────────────────────────────────
st.subheader("At a Glance")

SUMMARY_FNS = [
    ("VIX",             get_vix),
    ("Yield Curve",     get_yield_curve),
    ("HY Spread",       get_hy_credit_spread),
    ("DXY",             get_dxy),
    ("Consumer Conf.",  get_consumer_confidence),
]

cols = st.columns(len(SUMMARY_FNS))
for col, (name, fn) in zip(cols, SUMMARY_FNS):
    try:
        ind = fn()
        sc  = _score(ind)
        with col:
            st.metric(
                label=f"{COLOR_EMOJI[sc['color']]} {ind['label']}",
                value=f"{ind['current']:.2f} {ind['unit']}",
            )
            st.caption(sc["label"])
    except Exception as e:
        with col:
            st.metric(label=name, value="—")
            st.caption(f"Error: {e}")

st.divider()

# ─── Fear & Sentiment ─────────────────────────────────────────────────────────
st.header("Fear & Sentiment")

c1, c2, c3 = st.columns(3)
for col, fn in [(c1, get_vix), (c2, get_aaii_sentiment), (c3, get_consumer_confidence)]:
    try:
        ind = fn()
        sc  = _score(ind)
        _indicator_block(col, ind, sc)
    except Exception as e:
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

# VIX Term Structure
try:
    ts = get_vix_term_structure()
    st.markdown(f"**VIX Term Structure:** {ts['structure']}")
    if ts["vix9d"] and ts["vix"] and ts["vix3m"]:
        fig_ts = go.Figure(go.Bar(
            x=["VIX9D (9-day)", "VIX (30-day)", "VIX3M (3-month)"],
            y=[ts["vix9d"], ts["vix"], ts["vix3m"]],
            marker_color=["#EF553B" if ts["vix9d"] > ts["vix3m"] else "#00CC96"] * 3,
        ))
        fig_ts.update_layout(
            height=200, margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="VIX Level", showlegend=False,
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        if "Backwardation" in ts["structure"]:
            st.warning("VIX curve is in **backwardation** — short-term fear is elevated above long-term. A sign of near-term panic.")
        else:
            st.success("VIX curve is in **contango** — normal structure, near-term calm relative to longer horizon.")
except Exception as e:
    st.caption(f"VIX term structure unavailable: {e}")

st.divider()

# ─── Risk Appetite ────────────────────────────────────────────────────────────
st.header("Risk Appetite")

c4, c5, c6, c7 = st.columns(4)
for col, fn in [
    (c4, get_hy_credit_spread),
    (c5, get_gold_spy_ratio),
    (c6, get_dxy),
    (c7, get_copper_gold_ratio),
]:
    try:
        ind = fn()
        sc  = _score(ind)
        _indicator_block(col, ind, sc)
    except Exception as e:
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

st.divider()

# ─── Macro Stress ─────────────────────────────────────────────────────────────
st.header("Macro Stress")

c8, c9 = st.columns(2)
for col, fn in [(c8, get_yield_curve), (c9, get_fed_funds)]:
    try:
        ind = fn()
        sc  = _score(ind)
        note = None
        if "Yield Curve" in ind["label"]:
            if ind["current"] < 0:
                note = f"Curve is **inverted** ({ind['current']:.2f}%). Historically precedes recession by 12–18 months."
            elif ind["current"] < 0.5:
                note = f"Curve is near-flat ({ind['current']:.2f}%). Watch for inversion."
            else:
                note = f"Curve is positive ({ind['current']:.2f}%). Normal — longer rates above short-term."
        with col:
            st.metric(
                label=f"{COLOR_EMOJI[sc['color']]} {ind['label']}",
                value=f"{ind['current']:.2f} {ind['unit']}",
            )
            st.caption(f"{sc['label']} · {sc['score']:.0f}th pct")
            if note:
                st.info(note)
            fig = _sparkline(ind, sc, height=200)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

st.divider()

# ─── Liquidity ────────────────────────────────────────────────────────────────
st.header("Liquidity")

c10, c11 = st.columns(2)
for col, fn in [(c10, get_m2), (c11, get_fed_balance_sheet)]:
    try:
        ind = fn()
        sc  = _score(ind)
        _indicator_block(col, ind, sc)
    except Exception as e:
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

st.divider()

# ─── Advanced International ───────────────────────────────────────────────────
st.header("International & FX")

try:
    fx_df = get_fx_pairs()
    if not fx_df.empty:
        display_fx = fx_df.copy()
        display_fx["1d"] = display_fx["1d"].apply(_color_pct)
        display_fx["1w"] = display_fx["1w"].apply(_color_pct)
        display_fx = display_fx.rename(columns={"label": "Pair", "current": "Rate", "1d": "1 Day", "1w": "1 Week"})
        display_fx = display_fx.drop(columns=["ticker"])
        st.markdown("**FX Pairs**")
        st.write(display_fx.to_html(escape=False, index=False), unsafe_allow_html=True)
except Exception as e:
    st.caption(f"FX data unavailable: {e}")

st.divider()

# ─── Sector Rotation ──────────────────────────────────────────────────────────
st.header("Sector Rotation")
st.caption("SPDR sector ETF performance. Risk-On sectors rising = growth appetite. Risk-Off rising = defensive positioning.")

try:
    sec_df = get_sector_performance()
    if not sec_df.empty:
        ron = sec_df[sec_df["risk_type"] == "Risk-On"].copy()
        roff = sec_df[sec_df["risk_type"] == "Risk-Off"].copy()

        col_ron, col_roff = st.columns(2)
        for col, df, title in [(col_ron, ron, "Risk-On"), (col_roff, roff, "Risk-Off")]:
            with col:
                st.markdown(f"**{title}**")
                display = df.copy()
                for c in ["1d", "1w", "1m"]:
                    display[c] = display[c].apply(_color_pct)
                display = display.rename(columns={"label": "Sector", "1d": "1 Day", "1w": "1 Week", "1m": "1 Month"})
                display = display.drop(columns=["ticker", "risk_type"])
                st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)
except Exception as e:
    st.error(f"Sector data unavailable: {e}")

st.divider()

# ─── Global Markets ───────────────────────────────────────────────────────────
st.header("Global Markets")
st.caption("1-day, 1-week, and 1-month performance. Green = positive, red = negative.")

try:
    heatmap_df = get_global_heatmap()
    if heatmap_df.empty:
        st.info("No heatmap data available.")
    else:
        display = heatmap_df.copy()
        for c in ["1d", "1w", "1m"]:
            display[c] = display[c].apply(_color_pct)
        display = display.rename(columns={"label": "Market", "1d": "1 Day", "1w": "1 Week", "1m": "1 Month"})
        display = display.drop(columns=["ticker"])
        st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load global market data: {e}")
```

- [ ] **Step 2: Run full test suite one final time**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit and push**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add pages/2_Market_Pulse.py && git commit -m "feat: Market Pulse Phase 2b UI — Fear & Greed gauge, VIX structure, liquidity, FX, sectors"
cd /Users/akhilm/Claude-Projects/influx-lab && git push origin main
```

---

## Done: Phase 2b Complete

At this point:
- ✅ Fear & Greed composite — gauge dial with 6-indicator breakdown
- ✅ VIX term structure — contango vs backwardation with bar chart
- ✅ Liquidity — M2 + Fed balance sheet sparklines
- ✅ FX pairs — EUR/USD, USD/JPY, USD/CNY with 1d/1w returns
- ✅ Copper/Gold ratio — growth vs fear signal
- ✅ Sector rotation — Risk-On vs Risk-Off breakdown

**Deferred to Phase 2c:** Market breadth (500-ticker fetch), COT report (CFTC parsing), Insider transactions (EDGAR).

**Next plan:** `2026-03-25-phase3-integration.md` — cross-asset correlations, macro regime detection, lead-lag relationships.
