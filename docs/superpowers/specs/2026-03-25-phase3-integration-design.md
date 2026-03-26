---
title: Phase 3 — Integration (Market Relationship Lab)
date: 2026-03-25
status: approved
---

## Goal

Turn the individual Phase 2 indicators into a relationship analysis layer. The Integration module answers three questions the Market Pulse module can't:

1. **How do these assets move relative to each other?** (correlations)
2. **What kind of market environment are we in, and what has historically happened in that environment?** (regimes)
3. **Which indicators tend to move before others?** (lead-lag)

This is a hypothesis generator, not a prediction engine. The output is questions worth investigating, not signals to act on directly.

---

## Design Principles (inherited + extended)

- Every output answers "so what" — not just a number, but what it means
- Plain-language callouts throughout — no statistics background assumed
- Consistent visual language with Phase 2 (same five-band colors, same green/yellow/red scheme)
- Daily-frequency assets only for correlation and lead-lag math (weekly/monthly series are unsuitable for lag analysis)

---

## Data Constraints

Not all Phase 2 indicators are daily. The following split applies:

**Daily (usable for correlations + lead-lag):**
- VIX (`^VIX`)
- Dollar Index DXY (`DX-Y.NYB`)
- HY Credit Spread (`BAMLH0A0HYM2`) — daily FRED release
- Yield Curve 10Y–2Y (computed from `DGS2`, `DGS10`)
- Gold/SPY Ratio (`GLD`/`SPY`)
- Copper/Gold Ratio (`HG=F`/`GC=F`)
- Sector ETFs (`XLK`, `XLY`, `XLF`, `XLI`, `XLE`, `XLP`, `XLU`, `XLRE`, `XLV`, `XLB`)
- Global Indices (`SPY`, `^FTSE`, `^GDAXI`, `^N225`, `^HSI`, `EEM`)

**Weekly/Monthly (regime context only, not correlation math):**
- AAII Bull-Bear Spread (weekly)
- Consumer Confidence / UMich (monthly)
- M2 Money Supply (monthly)
- Fed Balance Sheet (weekly)
- Fed Funds Rate (monthly)

---

## Module Architecture

```
modules/integration/
├── __init__.py
├── correlations.py     ← rolling pairwise correlation matrix
├── regimes.py          ← regime classification + per-regime asset stats
└── leadlag.py          ← cross-correlation scanner across all indicators

tests/integration/
├── test_correlations.py
├── test_regimes.py
└── test_leadlag.py

pages/
└── 3_Integration.py    ← replace stub with three-section UI
```

---

## Section 1: Cross-Asset Correlations

### What it does
Computes a rolling pairwise Pearson correlation matrix across all daily-frequency assets. Renders as a color-coded heatmap.

### Interface
- Window selector at top: 30 / 60 / 90 days (default 60)
- Regime filter toggle: "All periods" or filter to a specific regime (Extreme Greed / Greed / Neutral / Fear / Extreme Fear). When a regime is selected, only the dates classified as that regime are included in the correlation calculation.
- Heatmap: assets on both axes, color scale deep green (+1.0) → white (0) → deep red (-1.0). Diagonal is always 1.0.
- Caption below: plain-language note explaining what correlation means and why regime-conditional correlations matter ("In a crisis, assets that normally move independently tend to move together — that's when diversification fails.")

### `correlations.py` interface
```python
def get_correlation_matrix(assets: list[str], window: int, regime_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    # Returns square DataFrame of Pearson correlations.
    # If regime_dates (DatetimeIndex) provided, filter price history to ONLY those dates before computing.
    # NaN values (missing prices or filtered-out dates) are dropped before Pearson calculation.
```

### Testing focus
- Returns square symmetric matrix
- Diagonal is 1.0
- Values bounded [-1, 1]
- Regime filter correctly subsets dates
- Handles missing data gracefully (drops NaN before computing)

---

## Section 2: Macro Regime

### What it does
Uses the Fear & Greed composite score history to classify each trading day into one of five regimes. Displays a color-coded timeline and per-regime asset behavior statistics.

### Regime bands (matching Phase 2b gauge)

| Score | Label | Color |
|-------|-------|-------|
| 0–20 | Extreme Greed | #00CC96 (green) |
| 20–40 | Greed | #7FBA00 (light green) |
| 40–60 | Neutral | #FFA500 (orange) |
| 60–80 | Fear | #FF6B35 (light red) |
| 80–100 | Extreme Fear | #EF553B (red) |

### Interface
1. **Current regime banner** — large colored banner at top: "Current Regime: Fear" with a one-sentence description of what that means
2. **Regime timeline** — horizontal color-coded strip across 3 years of F&G history. Each segment colored by regime. Plotly chart with hover showing exact score and date.
3. **Per-regime asset stats table** — for each regime, shows the average daily return of key assets (VIX, DXY, SPY, GLD, HY spread, Copper/Gold) while in that regime, computed from historical data. Helps answer "what tends to happen when we're in Fear?"

### Historical F&G Score Series

`compute_fear_greed_index()` returns only the current score. For the regime timeline we need scores across 3 years. A new function `get_fear_greed_history()` is added to `regimes.py`:

```python
def get_fear_greed_history() -> pd.Series:
    # Fetch the 6 composite indicator series (same as compute_fear_greed_index).
    # For each trading date in the common index, compute the equal-weighted average
    # percentile score across available indicators.
    # Returns pd.Series indexed by date, values 0–100.
    # Days where fewer than 3 indicators have data are excluded (NaN).
```

This is computed once and cached locally (same cache layer used by Phase 2 indicators).

### `regimes.py` interface
```python
def get_fear_greed_history() -> pd.Series:
    # See above — returns daily F&G composite score over 3-year window

def classify_regime(score: float) -> str:
    # Returns regime label for a single score

def get_regime_history(fg_scores: pd.Series) -> pd.Series:
    # Maps a series of F&G scores to regime labels, same index

def get_regime_asset_stats(regime_history: pd.Series, asset_prices: pd.DataFrame) -> pd.DataFrame:
    # For each regime, compute mean daily log return of each asset on those dates only.
    # Returns DataFrame: regime × asset → mean daily return.
    # If a regime has fewer than 5 observations, return NaN for that cell.
```

### Testing focus
- `classify_regime` returns correct label at all boundaries (0, 20, 40, 60, 80, 100)
- `get_regime_history` preserves index, maps all values
- `get_regime_asset_stats` returns one row per regime, one column per asset
- Handles missing asset data gracefully

---

## Section 3: Lead-Lag Scanner

### What it does
For a user-selected target series, scans all daily-frequency indicators and computes the cross-correlation at lags from -60 to +60 trading days. Surfaces which indicators have the strongest leading or lagging relationship with the target.

A positive lag means the indicator leads the target (the indicator moves first). A negative lag means the indicator lags the target (it follows after).

### Asset Universe

**Target options (7):** VIX, SPY, DXY, HY Spread, Yield Curve, Gold/SPY Ratio, Copper/Gold Ratio

**Indicator scan pool:** all daily-frequency assets minus the selected target:
- VIX, SPY, DXY, HY Spread, Yield Curve, Gold/SPY Ratio, Copper/Gold Ratio
- Sector ETFs: XLK, XLY, XLF, XLI, XLE, XLP, XLU, XLRE, XLV, XLB
- Global Indices: ^FTSE, ^GDAXI, ^N225, ^HSI, EEM

### Interface
1. **Target selector** — dropdown: pick from the 7 target options above
2. **Scan button** — runs cross-correlation of all other indicators vs target
3. **Results table** — ranked by absolute peak correlation: indicator name, peak correlation value, lag in days, plain-language interpretation ("Yield Curve leads VIX by ~45 days")
4. **Detail chart** — click any row → bar chart of correlation at every lag for that pair, with a vertical line at lag=0 and the peak highlighted

### `leadlag.py` interface
```python
def cross_correlate(series_a: pd.Series, series_b: pd.Series, max_lag: int = 60) -> pd.Series:
    # Computes Pearson correlation of series_a[t] vs series_b[t+lag] for lag in [-max_lag, +max_lag].
    # Positive lag: series_a LEADS series_b (a moves first, b follows).
    # Negative lag: series_a LAGS series_b (b moves first, a follows).
    # Series are aligned on common dates before computation.
    # Returns Series indexed by integer lag, values are Pearson correlation.

def scan_all_vs_target(target: pd.Series, indicators: dict[str, pd.Series], max_lag: int = 60) -> pd.DataFrame:
    # For each indicator, calls cross_correlate(indicator, target, max_lag).
    # Returns DataFrame sorted by abs(peak_correlation) descending:
    # columns: indicator_name, peak_correlation, peak_lag, interpretation
    # interpretation: e.g. "Yield Curve leads VIX by 45 days" or "XLK lags SPY by 3 days"
```

### Testing focus
- `cross_correlate` returns correct index range
- Lag=0 value matches standard Pearson correlation of the two series
- `scan_all_vs_target` returns one row per indicator, sorted by abs(peak_correlation) descending
- `interpret_lag` produces correct plain-language string for positive and negative lags
- Handles series with different lengths (align on common dates)

---

## Error Handling

All three sections wrap data loading in try/except with graceful degradation — a section that fails to load shows a warning, not a crash. This matches the Phase 2 pattern throughout the app.

---

## What This Is Not

- Not a prediction engine. Lead-lag patterns are historical tendencies, not guarantees.
- Not a trading signal generator. This is a research and hypothesis tool.
- Not exhaustive. Phase 3 uses daily-frequency assets only. Breadth, options positioning, COT, and insider data are deferred to later phases.

---

## Future Phases (noted, not built)

Phase 4 (Features): signal generation from the relationships discovered here — turning hypotheses into quantified, testable signals.

Phase 5 (ML/AI): pattern detection across the feature set, regime-conditional predictive models.

Additions deferred from Phase 3: volatility regime indicators, market breadth (500-ticker), options positioning (put/call ratio), COT report parsing.
