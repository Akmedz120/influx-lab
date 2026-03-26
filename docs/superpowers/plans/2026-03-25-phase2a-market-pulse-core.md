# Phase 2a: Market Pulse Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Market Pulse module with 8 core indicators (VIX, AAII sentiment, consumer confidence, HY credit spread, Gold/SPY ratio, DXY, yield curve, fed funds rate) plus a global market heatmap — all color-coded green/yellow/red against 3-year trailing history.

**Architecture:** Business logic lives in `modules/market_pulse/`. `scoring.py` provides a shared percentile ranking utility. `indicators.py` fetches and computes each indicator via the existing data layer (`modules/data/fetcher.py`). The UI page (`pages/2_Market_Pulse.py`) is a thin wrapper. `Home.py` gets a 5-indicator live summary strip.

**Tech Stack:** Python 3.11+, Streamlit, yfinance, fredapi, pandas, numpy, scipy, plotly, python-dotenv, pytest

---

## File Map

```
modules/market_pulse/
├── __init__.py                  ← empty
├── indicators.py                ← fetch + compute all Market Pulse indicators
└── scoring.py                   ← percentile_score utility

tests/market_pulse/
├── __init__.py                  ← empty
├── test_scoring.py
└── test_indicators.py

pages/
└── 2_Market_Pulse.py            ← replace stub with full UI

Home.py                          ← add live summary strip (modify existing)
```

**Color-coding convention (applies to all indicators):**
- `invert=False` — high value = stressed = red (VIX, HY spread, DXY, gold/SPY ratio, fed funds)
- `invert=True` — high value = calm = green (yield curve, AAII bull-bear spread, consumer confidence)
- Green = below 25th percentile · Yellow = 25th–75th · Red = above 75th

---

## Task 1: Module Scaffold

**Files:**
- Create: `modules/market_pulse/__init__.py`
- Create: `tests/market_pulse/__init__.py`

- [ ] **Step 1: Create empty init files**

Create `modules/market_pulse/__init__.py` (empty file).

Create `tests/market_pulse/__init__.py` (empty file).

- [ ] **Step 2: Verify pytest discovers the new test directory**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/ -v
```

Expected: `no tests ran` — no error, just empty collection.

- [ ] **Step 3: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/__init__.py tests/market_pulse/__init__.py && git commit -m "feat: scaffold market_pulse module"
```

---

## Task 2: Scoring Module

**Files:**
- Create: `modules/market_pulse/scoring.py`
- Create: `tests/market_pulse/test_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/market_pulse/test_scoring.py`:

```python
import pytest
import pandas as pd
from modules.market_pulse.scoring import percentile_score


@pytest.fixture
def uniform_series():
    return pd.Series(range(0, 101))  # 0-100, makes percentile math predictable


def test_high_value_is_stressed(uniform_series):
    result = percentile_score(uniform_series, 90.0)
    assert result["color"] == "red"
    assert result["label"] == "Stressed"


def test_low_value_is_calm(uniform_series):
    result = percentile_score(uniform_series, 10.0)
    assert result["color"] == "green"
    assert result["label"] == "Calm"


def test_middle_value_is_neutral(uniform_series):
    result = percentile_score(uniform_series, 50.0)
    assert result["color"] == "yellow"
    assert result["label"] == "Neutral"


def test_invert_flips_green_and_red(uniform_series):
    # Without invert: 90th pct = red. With invert: 90th pct = green.
    result = percentile_score(uniform_series, 90.0, invert=True)
    assert result["color"] == "green"
    assert result["label"] == "Calm"


def test_empty_series_returns_neutral():
    result = percentile_score(pd.Series([], dtype=float), 5.0)
    assert result["color"] == "yellow"
    assert result["score"] == 50.0


def test_returns_required_keys(uniform_series):
    result = percentile_score(uniform_series, 50.0)
    assert all(k in result for k in ["score", "color", "label"])


def test_score_is_float(uniform_series):
    result = percentile_score(uniform_series, 50.0)
    assert isinstance(result["score"], float)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_scoring.py -v
```

Expected: `ModuleNotFoundError` — scoring.py doesn't exist yet.

- [ ] **Step 3: Implement scoring.py**

Create `modules/market_pulse/scoring.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_scoring.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/scoring.py tests/market_pulse/test_scoring.py && git commit -m "feat: add percentile scoring utility for market pulse indicators"
```

---

## Task 3: Macro Stress Indicators (yield curve, fed funds)

This task creates `indicators.py` with the two macro stress functions.

**Files:**
- Create: `modules/market_pulse/indicators.py`
- Create: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Write failing tests**

Create `tests/market_pulse/test_indicators.py`:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


def make_series(n=500, value=5.0, noise=0.1):
    """Helper: make a time-indexed series of n values around `value`."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(value, noise, n), index=idx)


# ─── Macro Stress ─────────────────────────────────────────────────────────────

def test_get_yield_curve_returns_dict_keys():
    from modules.market_pulse.indicators import get_yield_curve
    dgs2  = make_series(500, value=4.0, noise=0)
    dgs10 = make_series(500, value=4.5, noise=0)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[dgs2, dgs10]):
        result = get_yield_curve()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_yield_curve_spread_is_10y_minus_2y():
    from modules.market_pulse.indicators import get_yield_curve
    dgs2  = make_series(100, value=4.0, noise=0)
    dgs10 = make_series(100, value=4.5, noise=0)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[dgs2, dgs10]):
        result = get_yield_curve()
    assert abs(result["current"] - 0.5) < 0.01


def test_get_yield_curve_invert_is_true():
    from modules.market_pulse.indicators import get_yield_curve
    s = make_series(100)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[s, s]):
        result = get_yield_curve()
    assert result["invert"] is True


def test_get_fed_funds_returns_dict_keys():
    from modules.market_pulse.indicators import get_fed_funds
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=make_series(100)):
        result = get_fed_funds()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_fed_funds_current_is_last_value():
    from modules.market_pulse.indicators import get_fed_funds
    s = make_series(50, value=5.25, noise=0)
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=s):
        result = get_fed_funds()
    assert abs(result["current"] - 5.25) < 0.01
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py -v
```

Expected: `ModuleNotFoundError` — indicators.py doesn't exist yet.

- [ ] **Step 3: Implement indicators.py with macro stress functions**

Create `modules/market_pulse/indicators.py`:

```python
import pandas as pd
from datetime import datetime, timedelta
from modules.data.fetcher import fetch_prices, fetch_fred

_THREE_YEAR_DAYS = 1095  # ~3 calendar years


def _start() -> str:
    return (datetime.today() - timedelta(days=_THREE_YEAR_DAYS)).strftime("%Y-%m-%d")


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def get_yield_curve() -> dict:
    """
    Fetch 2Y and 10Y treasury yields (FRED: DGS2, DGS10), return 10Y–2Y spread.
    Positive = normal/calm, negative = inverted (recession warning).
    Score with invert=True (high spread = calm).
    """
    start, end = _start(), _today()
    dgs2  = fetch_fred("DGS2", start, end)
    dgs10 = fetch_fred("DGS10", start, end)
    spread = (dgs10 - dgs2).dropna()
    return {
        "series": spread,
        "current": float(spread.iloc[-1]),
        "label": "Yield Curve (10Y–2Y)",
        "unit": "%",
        "invert": True,
    }


def get_fed_funds() -> dict:
    """
    Fetch Fed Funds Rate (FRED: FEDFUNDS).
    High rate = tighter financial conditions = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    series = fetch_fred("FEDFUNDS", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Fed Funds Rate",
        "unit": "%",
        "invert": False,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add yield curve and fed funds indicators"
```

---

## Task 4: Fear & Sentiment Indicators (VIX, AAII, consumer confidence)

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── Fear & Sentiment ─────────────────────────────────────────────────────────

def test_get_vix_returns_dict_keys():
    from modules.market_pulse.indicators import get_vix
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({"^VIX": np.full(500, 18.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_vix_current_is_last_value():
    from modules.market_pulse.indicators import get_vix
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_df = pd.DataFrame({"^VIX": np.linspace(15.0, 25.0, 100)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_vix()
    assert abs(result["current"] - 25.0) < 0.01


def test_get_aaii_sentiment_returns_dict_keys():
    from modules.market_pulse.indicators import get_aaii_sentiment
    weekly_idx = pd.date_range("2023-01-01", periods=100, freq="W")
    bull = pd.Series(np.full(100, 40.0), index=weekly_idx)
    bear = pd.Series(np.full(100, 30.0), index=weekly_idx)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[bull, bear]):
        result = get_aaii_sentiment()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_aaii_sentiment_spread_is_bull_minus_bear():
    from modules.market_pulse.indicators import get_aaii_sentiment
    weekly_idx = pd.date_range("2023-01-01", periods=50, freq="W")
    bull = pd.Series(np.full(50, 40.0), index=weekly_idx)
    bear = pd.Series(np.full(50, 25.0), index=weekly_idx)
    with patch("modules.market_pulse.indicators.fetch_fred", side_effect=[bull, bear]):
        result = get_aaii_sentiment()
    assert abs(result["current"] - 15.0) < 0.01


def test_get_consumer_confidence_returns_dict_keys():
    from modules.market_pulse.indicators import get_consumer_confidence
    monthly = pd.Series(
        np.linspace(60, 80, 36),
        index=pd.date_range("2023-01-01", periods=36, freq="ME")
    )
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=monthly):
        result = get_consumer_confidence()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_consumer_confidence_invert_is_true():
    from modules.market_pulse.indicators import get_consumer_confidence
    s = pd.Series(np.linspace(60, 80, 36), index=pd.date_range("2023-01-01", periods=36, freq="ME"))
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=s):
        result = get_consumer_confidence()
    assert result["invert"] is True
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_vix_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_vix_current_is_last_value tests/market_pulse/test_indicators.py::test_get_aaii_sentiment_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_aaii_sentiment_spread_is_bull_minus_bear tests/market_pulse/test_indicators.py::test_get_consumer_confidence_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_consumer_confidence_invert_is_true -v
```

Expected: `ImportError` — functions not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
def get_vix() -> dict:
    """
    Fetch VIX equity fear index (yfinance: ^VIX).
    High = stressed. Score with invert=False.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["^VIX"], start, end)
    series = prices["^VIX"].dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "VIX",
        "unit": "pts",
        "invert": False,
    }


def get_aaii_sentiment() -> dict:
    """
    Fetch AAII bull/bear survey (FRED: AAIIBULL, AAIIBEAR), return bull-bear spread.
    Low spread = fearful retail = stressed. High spread = bullish = calm.
    Score with invert=True. Note: weekly series.
    """
    start, end = _start(), _today()
    bull = fetch_fred("AAIIBULL", start, end)
    bear = fetch_fred("AAIIBEAR", start, end)
    combined = pd.concat([bull.rename("bull"), bear.rename("bear")], axis=1).dropna()
    spread = (combined["bull"] - combined["bear"]).rename("spread")
    return {
        "series": spread,
        "current": float(spread.iloc[-1]),
        "label": "AAII Bull–Bear Spread",
        "unit": "%",
        "invert": True,
    }


def get_consumer_confidence() -> dict:
    """
    Fetch U of Michigan Consumer Sentiment (FRED: UMCSENT).
    High = confident = calm. Score with invert=True. Note: monthly series.
    """
    start, end = _start(), _today()
    series = fetch_fred("UMCSENT", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Consumer Confidence (UMich)",
        "unit": "index",
        "invert": True,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py -v
```

Expected: 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add VIX, AAII sentiment, consumer confidence indicators"
```

---

## Task 5: Risk Appetite Indicators (HY credit spread, Gold/SPY ratio, DXY)

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── Risk Appetite ────────────────────────────────────────────────────────────

def test_get_hy_credit_spread_returns_dict_keys():
    from modules.market_pulse.indicators import get_hy_credit_spread
    with patch("modules.market_pulse.indicators.fetch_fred", return_value=make_series(500, value=4.0)):
        result = get_hy_credit_spread()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_gold_spy_ratio_returns_dict_keys():
    from modules.market_pulse.indicators import get_gold_spy_ratio
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({"GLD": np.full(500, 180.0), "SPY": np.full(500, 450.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_gold_spy_ratio()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])


def test_get_gold_spy_ratio_value_is_gld_over_spy():
    from modules.market_pulse.indicators import get_gold_spy_ratio
    idx = pd.date_range("2023-01-01", periods=100, freq="B")
    mock_df = pd.DataFrame({"GLD": np.full(100, 180.0), "SPY": np.full(100, 450.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_gold_spy_ratio()
    assert abs(result["current"] - (180.0 / 450.0)) < 0.001


def test_get_dxy_returns_dict_keys():
    from modules.market_pulse.indicators import get_dxy
    idx = pd.date_range("2023-01-01", periods=500, freq="B")
    mock_df = pd.DataFrame({"DX-Y.NYB": np.full(500, 104.0)}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_dxy()
    assert all(k in result for k in ["series", "current", "label", "unit", "invert"])
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_hy_credit_spread_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_gold_spy_ratio_returns_dict_keys tests/market_pulse/test_indicators.py::test_get_gold_spy_ratio_value_is_gld_over_spy tests/market_pulse/test_indicators.py::test_get_dxy_returns_dict_keys -v
```

Expected: `ImportError` — functions not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
def get_hy_credit_spread() -> dict:
    """
    Fetch US HY option-adjusted spread (FRED: BAMLH0A0HYM2).
    High spread = risky borrowing is expensive = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    series = fetch_fred("BAMLH0A0HYM2", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "HY Credit Spread",
        "unit": "%",
        "invert": False,
    }


def get_gold_spy_ratio() -> dict:
    """
    Compute GLD / SPY price ratio (yfinance: GLD, SPY).
    High ratio = flight to gold = risk-off = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["GLD", "SPY"], start, end)
    ratio = (prices["GLD"] / prices["SPY"]).dropna()
    return {
        "series": ratio,
        "current": float(ratio.iloc[-1]),
        "label": "Gold / SPY Ratio",
        "unit": "ratio",
        "invert": False,
    }


def get_dxy() -> dict:
    """
    Fetch Dollar Index (yfinance: DX-Y.NYB).
    Strong dollar = global risk-off = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["DX-Y.NYB"], start, end)
    series = prices["DX-Y.NYB"].dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Dollar Index (DXY)",
        "unit": "index",
        "invert": False,
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py -v
```

Expected: 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add HY credit spread, Gold/SPY ratio, DXY indicators"
```

---

## Task 6: Global Market Heatmap

**Files:**
- Modify: `modules/market_pulse/indicators.py`
- Modify: `tests/market_pulse/test_indicators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/market_pulse/test_indicators.py`:

```python
# ─── Global Heatmap ───────────────────────────────────────────────────────────

def test_get_global_heatmap_returns_dataframe():
    from modules.market_pulse.indicators import get_global_heatmap, HEATMAP_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({t: np.linspace(100, 110, 30) for t in HEATMAP_TICKERS}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_global_heatmap()
    assert isinstance(result, pd.DataFrame)


def test_get_global_heatmap_has_required_columns():
    from modules.market_pulse.indicators import get_global_heatmap, HEATMAP_TICKERS
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({t: np.linspace(100, 110, 30) for t in HEATMAP_TICKERS}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_global_heatmap()
    assert all(c in result.columns for c in ["ticker", "label", "1d", "1w", "1m"])


def test_get_global_heatmap_1d_return_correct():
    from modules.market_pulse.indicators import get_global_heatmap
    prices = np.linspace(100.0, 110.0, 30)
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    mock_df = pd.DataFrame({"SPY": prices}, index=idx)
    with patch("modules.market_pulse.indicators.fetch_prices", return_value=mock_df):
        result = get_global_heatmap()
    spy_row = result[result["ticker"] == "SPY"].iloc[0]
    expected_1d = prices[-1] / prices[-2] - 1
    assert abs(spy_row["1d"] - expected_1d) < 1e-6
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/test_indicators.py::test_get_global_heatmap_returns_dataframe tests/market_pulse/test_indicators.py::test_get_global_heatmap_has_required_columns tests/market_pulse/test_indicators.py::test_get_global_heatmap_1d_return_correct -v
```

Expected: `ImportError` — `HEATMAP_TICKERS` and `get_global_heatmap` not defined yet.

- [ ] **Step 3: Append to indicators.py**

Append to `modules/market_pulse/indicators.py`:

```python
HEATMAP_TICKERS = ["SPY", "^FTSE", "^GDAXI", "^N225", "^HSI", "EEM"]
HEATMAP_LABELS = {
    "SPY":    "S&P 500 (US)",
    "^FTSE":  "FTSE 100 (UK)",
    "^GDAXI": "DAX (Germany)",
    "^N225":  "Nikkei 225 (Japan)",
    "^HSI":   "Hang Seng (HK)",
    "EEM":    "Emerging Markets",
}


def get_global_heatmap() -> pd.DataFrame:
    """
    Fetch 40-day price history for global indices (yfinance).
    Returns DataFrame: ticker, label, 1d return, 1w return, 1m return.
    Uses a 40-day window — enough for 1-month + buffer on short weeks/holidays.
    """
    end = _today()
    start = (datetime.today() - timedelta(days=40)).strftime("%Y-%m-%d")
    prices = fetch_prices(HEATMAP_TICKERS, start, end)

    rows = []
    for ticker in HEATMAP_TICKERS:
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
            "label": HEATMAP_LABELS.get(ticker, ticker),
            "1d": r1d,
            "1w": r1w,
            "1m": r1m,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run full indicator test suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/market_pulse/ -v
```

Expected: 25 tests PASS (7 scoring + 18 indicator).

- [ ] **Step 5: Run full project test suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add modules/market_pulse/indicators.py tests/market_pulse/test_indicators.py && git commit -m "feat: add global market heatmap indicator"
```

---

## Task 7: Market Pulse UI Page

**Files:**
- Modify: `pages/2_Market_Pulse.py` (replace stub)

No unit tests — UI layer only calls tested module functions.

- [ ] **Step 1: Replace pages/2_Market_Pulse.py**

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.market_pulse.indicators import (
    get_yield_curve, get_fed_funds, get_vix,
    get_aaii_sentiment, get_consumer_confidence,
    get_hy_credit_spread, get_gold_spy_ratio, get_dxy,
    get_global_heatmap,
)
from modules.market_pulse.scoring import percentile_score

st.set_page_config(page_title="Market Pulse", layout="wide")
st.title("Market Pulse")
st.caption("Daily read on market mood and stress — color-coded green/yellow/red against 3-year history.")

COLOR_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
COLOR_HEX   = {"green": "#00CC96", "yellow": "#FFA500", "red": "#EF553B"}


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


# ─── Summary Strip ────────────────────────────────────────────────────────────
st.subheader("At a Glance")

SUMMARY_FNS = [
    ("VIX",                get_vix),
    ("Yield Curve",        get_yield_curve),
    ("HY Spread",          get_hy_credit_spread),
    ("DXY",                get_dxy),
    ("Consumer Conf.",     get_consumer_confidence),
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

st.divider()

# ─── Risk Appetite ────────────────────────────────────────────────────────────
st.header("Risk Appetite")

c4, c5, c6 = st.columns(3)
for col, fn in [(c4, get_hy_credit_spread), (c5, get_gold_spy_ratio), (c6, get_dxy)]:
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

c7, c8 = st.columns(2)
for col, fn in [(c7, get_yield_curve), (c8, get_fed_funds)]:
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

# ─── Global Markets ───────────────────────────────────────────────────────────
st.header("Global Markets")
st.caption("1-day, 1-week, and 1-month performance. Green = positive, red = negative.")

try:
    heatmap_df = get_global_heatmap()
    if heatmap_df.empty:
        st.info("No heatmap data available.")
    else:
        def _color_pct(val):
            if val is None or pd.isna(val):
                return "—"
            color = "#00CC96" if val >= 0 else "#EF553B"
            return f'<span style="color:{color}">{val:.2%}</span>'

        display = heatmap_df.copy()
        for col in ["1d", "1w", "1m"]:
            display[col] = display[col].apply(_color_pct)
        display = display.rename(columns={"label": "Market", "1d": "1 Day", "1w": "1 Week", "1m": "1 Month"})
        display = display.drop(columns=["ticker"])
        st.write(display.to_html(escape=False, index=False), unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load global market data: {e}")
```

- [ ] **Step 2: Verify app runs**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/streamlit run Home.py
```

Navigate to Market Pulse. Confirm: all 4 sections render (or show graceful errors if no FRED API key).

- [ ] **Step 3: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add pages/2_Market_Pulse.py && git commit -m "feat: Market Pulse UI — fear/sentiment, risk appetite, macro stress, global heatmap"
```

---

## Task 8: Update Home.py — Live Summary Strip

**Files:**
- Modify: `Home.py`

No unit tests — UI layer only.

- [ ] **Step 1: Replace Home.py**

```python
import streamlit as st
from modules.market_pulse.indicators import (
    get_vix, get_yield_curve, get_hy_credit_spread,
    get_dxy, get_consumer_confidence,
)
from modules.market_pulse.scoring import percentile_score

st.set_page_config(page_title="InFlux Lab", layout="wide")

st.title("InFlux Lab")
st.caption("A local research environment for financial modeling, signal development, and market analysis.")

# ─── Live Market Pulse Strip ──────────────────────────────────────────────────
COLOR_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

SUMMARY = [
    ("VIX",             get_vix),
    ("Yield Curve",     get_yield_curve),
    ("HY Spread",       get_hy_credit_spread),
    ("DXY",             get_dxy),
    ("Consumer Conf.",  get_consumer_confidence),
]

cols = st.columns(len(SUMMARY))
for col, (name, fn) in zip(cols, SUMMARY):
    with col:
        try:
            ind = fn()
            sc  = percentile_score(ind["series"], ind["current"], invert=ind.get("invert", False))
            st.metric(
                label=f"{COLOR_EMOJI[sc['color']]} {ind['label']}",
                value=f"{ind['current']:.2f} {ind['unit']}",
            )
            st.caption(sc["label"])
        except Exception:
            st.metric(label=name, value="—")
            st.caption("Unavailable")

st.divider()

# ─── Module Overview ──────────────────────────────────────────────────────────
st.markdown("""
This workspace is organized into focused modules. Each module builds on the last.
Use the sidebar to navigate.

| Module | Purpose | Status |
|--------|---------|--------|
| **Foundations** | Returns, distributions, Monte Carlo simulation | ✅ Active |
| **Market Pulse** | Sentiment, fear/greed, risk appetite, macro stress | ✅ Active |
| **Integration** | Cross-asset correlations, macro regime analysis | 🔜 Coming |
| **Features** | Signal generation, derived metrics | 🔜 Coming |
| **ML / AI** | Pattern detection, predictive models on structured features | 🔜 Coming |
| **Sectors** | Sector rotation, relative strength, money flow | 🔜 Coming |
| **Sandbox** | Free experimentation | 🔜 Coming |
""")
```

- [ ] **Step 2: Run final full test suite**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && .venv/bin/pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/akhilm/Claude-Projects/influx-lab && git add Home.py && git commit -m "feat: add live market pulse summary strip to home page"
```

---

## Done: Phase 2a Complete

At this point:
- ✅ `modules/market_pulse/scoring.py` — percentile scoring with green/yellow/red
- ✅ `modules/market_pulse/indicators.py` — 8 indicators + global heatmap
- ✅ `pages/2_Market_Pulse.py` — full UI with sparklines, metrics, and "so what" callouts
- ✅ `Home.py` — live 5-indicator summary strip
- ✅ All business logic covered by tests (39 Phase 1 + ~25 Phase 2a)

**Next plan:** `2026-03-25-phase2b-market-pulse-advanced.md` — volatility structure, market breadth, M2/Fed balance sheet, custom Fear & Greed composite.
