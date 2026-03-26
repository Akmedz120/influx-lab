---
title: Financial Research Workspace — Design Spec
date: 2026-03-25
status: approved
---

## Vision

A local financial research and modeling environment for personal use, built for learning, experimentation, and gradual development into more advanced analytical tools. Built for the user first, with product potential later. Every module produces actionable insights — not just charts and numbers.

## Interface

Streamlit multi-page app. Runs locally, accessed via browser at localhost. No deployment, no hosting, no accounts. Just run `streamlit run Home.py` and it opens in your browser.

## Design Principles

- **Modular**: each module is independent, can be understood and extended on its own
- **Actionable**: every output answers "so what" — connects data to decisions
- **Learning-friendly**: explanations baked into each section explaining why it matters
- **Clean architecture**: product-ready from day one
- **Caching**: local file-based cache (pickle), 24-hour expiry for daily data, 1-hour for intraday
- **UI quality**: intuitive, smart, good infographics — prioritized after functionality is solid

## App Structure

```
Home.py                  ← Landing page: workspace overview + live market pulse summary
pages/
  1_Foundations.py       ← PHASE 1 ACTIVE: Returns, distributions, Monte Carlo
  2_Market_Pulse.py      ← PHASE 2: Sentiment, fear, risk appetite, international
  3_Integration.py       ← STUB: Cross-asset relationships
  4_Features.py          ← STUB: Signal generation
  5_ML_AI.py             ← STUB: Machine learning experiments
  6_Sectors.py           ← STUB: Sector rotation (future)
  7_Sandbox.py           ← STUB: Free experimentation
modules/
  data/                  ← Data fetching + caching layer
  foundations/           ← Foundations logic
  market_pulse/          ← Market Pulse logic
config.py                ← App-wide settings
```

Stub pages show module name, a one-sentence description of purpose, and a "not yet built" note. No fake content.

**Home page**: Static overview of what each module does + a live summary strip of 4-5 key Market Pulse indicators (VIX, yield curve, credit spread, DXY) once Market Pulse is built. In Phase 1, home page is text-only overview.

---

## Phase 1: Foundations Module

### Returns & Distributions

- Input: one or more tickers + date range, pulled via yfinance
- Calculate daily, weekly, monthly returns
- **Single asset view**: return series chart + distribution histogram with normal distribution overlaid — visually shows fat tails
- **Comparison view**: two assets on the same histogram (overlaid, semi-transparent) so differences in shape are immediately visible. Also side-by-side stats table.
- Key stats per asset: mean return, volatility (std dev), skewness, kurtosis
- "So what" explanation: dynamic — e.g. "SPY has negative skew of -0.8, meaning large losses occur more frequently than a normal model would predict"

### Monte Carlo Simulation

- **Method**: Geometric Brownian Motion (GBM) — standard starting point. Known limitation: assumes log-normal returns, which underestimates tail risk. Noted in the UI. Can be upgraded to historical simulation later.
- Simulate 1,000 possible future price paths for any ticker
- User sets time horizon (30 / 90 / 252 trading days)
- Show full fan of paths + highlighted median, 5th percentile (bad outcome), 95th percentile (good outcome)
- Key stats: median final price, probability of loss, worst 5% outcome
- "So what" explanation: "This shows the range of outcomes if history repeats. The wide fan means high uncertainty — not a prediction, a probability distribution."

### Design Note

Every section connects numbers to decisions. Raw stats alone are not enough. The standard is: number → what it means → why it matters for decisions.

---

## Phase 2: Market Pulse Module

A daily-use dashboard showing the mood and stress level of the market. Color-coded green/yellow/red.

**Color-coding methodology**: Each indicator is scored against its own trailing 3-year history using percentile thresholds. Green = below 25th percentile (calm), yellow = 25th–75th (neutral), red = above 75th percentile (stressed). Inverted for indicators where high = good (e.g. breadth).

Split into two sub-phases to keep scope manageable:

---

### Phase 2a — Core (build first)

The highest-signal, most accessible indicators. All available via yfinance or FRED with no complex parsing.

**Fear & Sentiment**
- VIX — equity market fear (yfinance: `^VIX`)
- AAII sentiment survey — retail bullish/bearish reading; contrarian signal (FRED: `AAIIBULL`, `AAIIBEAR`)
- Consumer confidence — University of Michigan (FRED: `UMCSENT`)

**Risk Appetite**
- High yield credit spread — risky borrowing cost vs safe (FRED: `BAMLH0A0HYM2`)
- Gold vs SPY ratio — safe haven demand vs equities (yfinance: `GLD`, `SPY`)
- Dollar index DXY — strong dollar = global risk-off (yfinance: `DX-Y.NYB`)

**Macro Stress**
- Yield curve (2yr/10yr spread) — inverted = recession warning (FRED: `DGS2`, `DGS10`)
- Fed funds rate (FRED: `FEDFUNDS`)

**International (core)**
- Global market heatmap: SPY, `^FTSE`, `^GDAXI`, `^N225`, `^HSI`, `EEM` — color-coded by 1d/1w/1m performance

---

### Phase 2b — Advanced (build after 2a is solid)

**Volatility Structure**
- Term structure: VIX9D / VIX / VIX3M — contango (calm) vs backwardation (panic)
- MOVE index — bond market VIX. *Note: verify free availability before building; may require alternative source.*

**Market Breadth**
- % of S&P 500 stocks above 200-day MA
- New 52-week highs vs lows
- *Data sourcing note*: requires fetching all S&P 500 components. Use a static list of current constituents (sourced from Wikipedia or a stored CSV) + yfinance batch fetch. This is more work than a single ticker pull — plan accordingly.

**Liquidity**
- M2 money supply trend (FRED: `M2SL`)
- Fed balance sheet size (FRED: `WALCL`)

**International (advanced)**
- Major FX pairs vs USD: EUR/USD, USD/JPY, USD/CNY
- Copper/Gold ratio — global growth vs fear (yfinance: `HG=F`, `GC=F`)
- EM stress: EEM performance + EM bond spread if available

**Smart Money / Institutional**
- COT Report — commercial vs speculator positioning. *Data sourcing note*: free from CFTC (weekly CSV releases). Parsing is non-trivial — allocate extra build time for this one.
- Insider buying/selling (SEC EDGAR). Use `edgartools` Python library to simplify access.

**Single Stock Pulse**
- For any ticker: implied vol vs historical vol (yfinance options chain)
- Insider transaction activity (EDGAR)
- *Short interest*: **known gap** — freely available short interest data is limited. Finviz or similar require paid access or scraping. Placeholder in UI for now; revisit when a clean free source is identified.

**Proprietary Composite**
- Custom Fear & Greed Index built from Phase 2a indicators
- **Methodology**: 6 core indicators (VIX, credit spread, DXY, yield curve, gold/SPY ratio, AAII sentiment), each normalized to a 0–100 scale using 3-year percentile rank, then equal-weighted average. Final score: 0 = extreme greed, 100 = extreme fear.
- Displayed as a gauge/dial — the daily gut-check number

**Sector Rotation (pulse view)**
- Quick read: which SPDR sector ETFs (XLF, XLK, XLE, XLV, XLI, XLY, XLP, XLU, XLRE, XLB) are gaining/losing flows
- Risk-on vs risk-off sector positioning
- *Note*: this is a high-level pulse only. Deep sector analysis lives in the dedicated Sectors module (6_Sectors.py)

---

## Data Layer

| Source | Covers | Notes |
|--------|--------|-------|
| yfinance | Stocks, ETFs, crypto, indices, VIX, DXY, commodities, options chains | Free, no API key |
| FRED | Macro: yield curve, CPI, Fed rate, M2, credit spreads, AAII, consumer confidence | Free — API key required (takes 2 min to get) |
| CFTC | COT report — institutional futures positioning | Free, weekly CSV — parsing work required |
| SEC EDGAR | Insider transactions via `edgartools` library | Free |
| File upload | Custom CSV data | For anything not available elsewhere |

**Caching**: `modules/data/cache.py` — pickle-based local cache. 24-hour TTL for daily data (prices, macro), 1-hour TTL for anything more frequent. Cache stored in `data/cache/`. Prevents re-fetching on every page load.

---

## Future Modules (Stubbed)

| Module | Purpose |
|--------|---------|
| Integration | Cross-asset correlations, macro regime analysis, lead-lag relationships between assets |
| Features | Derived signals: volatility states, momentum, regime classification, macro indicators as structured inputs |
| ML/AI | Pattern detection and predictive modeling — operates on structured features, not raw prices |
| Sectors | Deep sector rotation: relative strength, historical patterns, money flow analysis |
| Sandbox | Free experimentation — unconventional ideas, exploratory models, anything goes |

---

## Build Sequence

1. **Phase 1**: Skeleton (all stubs) + Foundations module fully built
2. **Phase 2a**: Market Pulse core indicators — VIX, sentiment, credit spreads, yield curve, DXY, global heatmap
3. **Phase 2b**: Market Pulse advanced — breadth, COT, insider, volatility structure, custom Fear & Greed
4. **Phase 3+**: Integration, Features, ML/AI in order — each built fully before moving to the next

**Definition of "done" for each phase**: module is functional, produces actionable output, has "so what" explanations, and doesn't break when data is unavailable.

---

## Known Gaps & Risks

| Issue | Impact | Status |
|-------|--------|--------|
| Short interest data not freely available | Single Stock Pulse incomplete | Placeholder — revisit |
| Breadth data requires 500 component fetches | Slow build, API rate limits | Use static constituent list + batch fetch |
| COT parsing complexity | Extra build time | Plan for it explicitly |
| MOVE index availability | May need alternative bond vol source | Verify before building |
| GBM underestimates tail risk | Monte Carlo less accurate in stress scenarios | Documented in UI, improvable later |
