---
title: Phase 4 — Features Module Design Spec
date: 2026-03-25
status: approved
---

## Goal

Build a feature engineering layer that turns raw market data into clean, labeled signals — useful as a daily visual dashboard and as structured input for future ML models.

## Design Principles

- **Incremental**: start with 9 well-understood signals; adding more is one function + one column
- **Digestible**: every signal has a human-readable label (low/mid/high or numeric) and a plain-English explanation
- **ML-ready**: all signals output clean numeric values in a date-indexed DataFrame
- **No new data sources**: reuses existing fetch/cache layer and Phase 3 macro modules
- **Progressively extensible**: Phase 5 builds on top — regime state machine, backtesting, GenAI narrative layer deferred

---

## Architecture

### New files

```
modules/features/
    __init__.py
    signals.py          ← 9 signal functions, one per signal
    feature_table.py    ← assembles signals into date × feature DataFrame

tests/features/
    __init__.py
    test_signals.py     ← unit tests for each signal function
    test_feature_table.py

pages/4_Features.py     ← full UI (replaces stub)
```

### No changes to existing modules

Macro Stress signal is computed by calling `get_fear_greed_history()` from `modules/integration/regimes` — no duplication.

---

## Signal Definitions

### Function signatures and return types

Each signal function returns a single `pd.Series` of numeric values indexed the same as the input. For signals that also need a categorical label, `feature_table.build()` applies a shared `_score_to_label(score, low=33, high=67)` helper to derive it — the signal function itself only returns the numeric score.

| # | Signal | Function signature | Return | Description |
|---|--------|-------------------|--------|-------------|
| 1 | Volatility Regime | `volatility_regime(prices: pd.Series) -> pd.Series` | 0–100 numeric | 20-day realized vol percentile-ranked against trailing 252-day window |
| 2 | Momentum Short | `momentum_short(prices: pd.Series) -> pd.Series` | numeric, signed | 20-day log return |
| 3 | Momentum Long | `momentum_long(prices: pd.Series) -> pd.Series` | numeric, signed | 90-day log return |
| 4 | Mean Reversion | `mean_reversion(prices: pd.Series) -> pd.Series` | numeric z-score | (price − 60d SMA) / 60d rolling std |
| 5 | Trend Strength | `trend_strength(prices: pd.Series) -> pd.Series` | 0–1 numeric | Rolling R² of OLS fit on 60-day windows of log prices |
| 6 | Macro Stress | `macro_stress(prices: pd.Series) -> pd.Series` | 0–100 numeric | Calls `get_fear_greed_history()`, reindexes to `prices.index` using forward-fill. Ignores prices values — the argument is used only to determine the output index. Returns NaN for dates outside the 3-year history window. |
| 7 | Volume Regime | `volume_regime(volume: pd.Series \| None) -> pd.Series` | 0–100 numeric | 20-day avg volume percentile-ranked against trailing 252-day window. If `volume` is `None` or empty, returns a Series of NaN with no index. |
| 8 | 52-Week Position | `fiftytwo_week_position(prices: pd.Series) -> pd.Series` | 0–100 numeric | (price − 252d rolling min) / (252d rolling max − 252d rolling min) × 100 |
| 9 | Relative Strength vs Benchmark | `relative_strength(prices: pd.Series, benchmark: pd.Series) -> pd.Series` | numeric, signed | Asset 20-day log return minus benchmark 20-day log return, aligned on common dates |

### Categorical thresholds (applied by `feature_table.build()`, not by signal functions)

`_score_to_label(score: float, low: float = 33, high: float = 67) -> str`
- score < 33 → `"low"`
- 33 ≤ score ≤ 67 → `"mid"`
- score > 67 → `"high"`

Applied to: Volatility Regime score, Volume Regime score.

---

## Feature Table

`feature_table.build(prices: pd.Series, volume: pd.Series | None = None, benchmark: pd.Series | None = None) -> pd.DataFrame`

- Computes all 9 signals for the given price series
- Returns a DataFrame indexed by `prices.index`, one column per signal
- Columns: `vol_regime_score`, `vol_regime_label`, `momentum_short`, `momentum_long`, `mean_reversion`, `trend_strength`, `macro_stress`, `volume_regime_score`, `volume_regime_label`, `fiftytwo_week_pos`, `relative_strength`
- If `benchmark` is None, `relative_strength` column is NaN
- If `volume` is None, `volume_regime_score` and `volume_regime_label` columns are NaN
- `vol_regime_label` and `volume_regime_label` are derived by applying `_score_to_label()` to the respective score columns

---

## Dashboard — `4_Features.py`

### Section 1: Signal Snapshot

**Ticker input** at top — any yfinance-compatible symbol (e.g. SPY, AAPL, GLD, ^VIX).

**Radar chart (Market Fingerprint)** — spider/radar chart with one axis per signal, normalized 0–1 for display. Normalization rules per signal type:
- Bounded 0–100 signals (`vol_regime_score`, `macro_stress`, `fiftytwo_week_pos`, `volume_regime_score`): divide by 100
- Already 0–1 (`trend_strength`): use as-is
- Signed unbounded signals (`momentum_short`, `momentum_long`, `mean_reversion`, `relative_strength`): clip to [−0.15, +0.15] for momentum, [−3, +3] for z-scores/RS, then scale linearly to [0, 1] with 0.5 = zero

Shows the "shape" of the asset's current state. Labels on each axis with the current value. Plotly `go.Scatterpolar`.

**Signal badge row** — below the radar chart, 9 colored badges (one per signal) showing label + value. Color: green (calm/positive), amber (neutral), red (stressed/extreme). Digestible in one glance.

**Divergence flag** — checks three specific conditions using the latest signal values. If any are true, show a visible amber banner with plain-English explanation. Not shown when signals are consistent.

Divergence conditions (applied to the most recent row of the feature table):
1. **Vol/Momentum divergence**: `vol_regime_label == "high"` AND `momentum_short > 0` AND `momentum_long > 0` — vol is elevated but prices are still rising
2. **Macro/Price divergence**: `macro_stress > 60` AND `momentum_short > 0` — macro stress is elevated (fear side) but short-term momentum is positive
3. **Momentum conflict**: `momentum_short` and `momentum_long` have opposite signs (one positive, one negative) — short-term and long-term trends disagree

Example banner text: "Divergence: Volatility regime is high but momentum is positive — stress and price trend are pointing in opposite directions."

### Section 2: Signal History

**Signal selector** — dropdown to pick any of the 9 signals.

**Line chart** — signal value over the last 2 years. Regime bands shaded in background for categorical signals (green band = low, amber = mid, red = high). Plotly `go.Figure`.

**Calendar heatmap** — below the line chart, a GitHub-style calendar grid for the selected signal over the last 1 year. Each day = one square, colored by signal intensity. Makes seasonal patterns and clusters immediately visible. Plotly `go.Heatmap` with rows = day of week (Mon=0 … Sun=6, 7 rows) and columns = week number of year (~53 columns). Cells without data are transparent.

### Section 3: Feature Table

**Data grid** — full date × feature DataFrame displayed with `st.dataframe`. Most recent dates at top.

**Download button** — `st.download_button` exports the table as CSV. Filename: `{ticker}_features_{date}.csv`.

**Row count caption** — "Showing N trading days of features."

---

## Error Handling

- If ticker not found by yfinance: show `st.error` with ticker name, do not crash page
- If macro stress data unavailable: `macro_stress` column is NaN, signal badge shows "—"
- If volume data unavailable (e.g. index like ^VIX): volume regime columns silently NaN, badge hidden
- All signal functions accept any length series ≥ 2; return NaN for dates with insufficient lookback

---

## Testing

### `tests/features/test_signals.py`

Each signal function is a pure function (series in → series out). Tests use synthetic price series — no API calls.

| Test | Assertion |
|------|-----------|
| `test_volatility_regime_returns_series` | Output index matches input index |
| `test_volatility_regime_labels_are_valid` | All non-NaN labels are in {low, mid, high} |
| `test_volatility_regime_high_vol_scores_high` | Constructed high-vol period scores > 67 |
| `test_momentum_short_positive_for_rising_prices` | Monotonically rising series → positive momentum |
| `test_momentum_short_negative_for_falling_prices` | Monotonically falling series → negative momentum |
| `test_momentum_long_uses_90d_window` | 90-day return matches manual calculation |
| `test_mean_reversion_zero_at_moving_average` | Price at exact SMA → z-score ≈ 0 |
| `test_mean_reversion_positive_above_sma` | Price above SMA → positive z-score |
| `test_trend_strength_high_for_linear_trend` | Linear price series → R² close to 1.0 |
| `test_trend_strength_low_for_random_walk` | Random walk → R² near 0 |
| `test_fiftytwo_week_position_at_high` | Price = 52w high → score = 100 |
| `test_fiftytwo_week_position_at_low` | Price = 52w low → score = 0 |
| `test_relative_strength_zero_when_equal` | Asset and benchmark same series → RS = 0 |
| `test_relative_strength_positive_when_outperforms` | Asset outperforms benchmark → positive RS |
| `test_volume_regime_nan_when_no_volume` | No volume series → NaN output |

### `tests/features/test_feature_table.py`

| Test | Assertion |
|------|-----------|
| `test_feature_table_has_required_columns` | All 11 expected columns present |
| `test_feature_table_index_matches_prices` | Output index = input price series index |
| `test_feature_table_handles_missing_benchmark` | `relative_strength` column is NaN when benchmark=None |

---

## What's Deferred to Phase 5

- Regime state machine (multi-dimensional state labels from combined signals)
- ML model training and evaluation
- Backtesting / conditional return analysis ("days like today")
- GenAI narrative layer (plain-English daily briefing)
- Niche / proprietary signal development
- Signal combination builder (user-defined composite scores)

---

## Build Sequence

1. Scaffold `modules/features/` and `tests/features/`
2. Build and test `signals.py` (9 signal functions)
3. Build and test `feature_table.py`
4. Build `4_Features.py` UI
5. Run full test suite, commit, push
