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
    "vol_regime_score":    "Volatility Regime",
    "momentum_short":      "Momentum (20d)",
    "momentum_long":       "Momentum (90d)",
    "mean_reversion":      "Mean Reversion",
    "trend_strength":      "Trend Strength",
    "macro_stress":        "Macro Stress",
    "volume_regime_score": "Volume Regime",
    "fiftytwo_week_pos":   "52-Week Position",
    "relative_strength":   "Relative Strength",
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

_RADAR_NORM = {
    "vol_regime_score":    ("bounded", 0,     100),
    "momentum_short":      ("signed",  -0.15, 0.15),
    "momentum_long":       ("signed",  -0.15, 0.15),
    "mean_reversion":      ("signed",  -3.0,  3.0),
    "trend_strength":      ("bounded", 0,     1),
    "macro_stress":        ("bounded", 0,     100),
    "volume_regime_score": ("bounded", 0,     100),
    "fiftytwo_week_pos":   ("bounded", 0,     100),
    "relative_strength":   ("signed",  -3.0,  3.0),
}

RADAR_SIGNALS = list(_RADAR_NORM.keys())


def _normalize_for_radar(value: float, signal: str) -> float:
    if pd.isna(value):
        return 0.5
    mode, lo, hi = _RADAR_NORM[signal]
    if mode == "bounded":
        return float(np.clip((value - lo) / (hi - lo), 0, 1))
    clipped = float(np.clip(value, lo, hi))
    return (clipped - lo) / (hi - lo)


def _badge_color(signal: str, value: float) -> str:
    if pd.isna(value):
        return "#888888"
    stressed_high = {"vol_regime_score", "macro_stress"}
    good_high     = {"fiftytwo_week_pos", "trend_strength",
                     "momentum_short", "momentum_long", "relative_strength"}
    norm = _normalize_for_radar(value, signal)
    if signal in stressed_high:
        if norm > 0.67: return "#EF553B"
        if norm > 0.33: return "#FFA500"
        return "#00CC96"
    if signal in good_high:
        if norm > 0.67: return "#00CC96"
        if norm > 0.33: return "#FFA500"
        return "#EF553B"
    # volume_regime and mean_reversion: extremes = amber
    if 0.33 <= norm <= 0.67: return "#00CC96"
    return "#FFA500"


def _check_divergences(row: pd.Series) -> list[str]:
    msgs = []
    vol_label = row.get("vol_regime_label", "mid")
    ms    = row.get("momentum_short", float("nan"))
    ml    = row.get("momentum_long",  float("nan"))
    macro = row.get("macro_stress",   float("nan"))

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
    # Prices
    try:
        prices_df = fetch_prices([ticker], start, end)
        prices = prices_df[ticker].dropna() if ticker in prices_df.columns else pd.Series(dtype=float)
    except Exception:
        prices = pd.Series(dtype=float)
    # Volume via yfinance Ticker (not in cached fetcher)
    volume = None
    try:
        hist = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        if "Volume" in hist.columns:
            volume = hist["Volume"].dropna()
            if volume.index.tz is not None:
                volume.index = volume.index.tz_convert(None)
    except Exception:
        pass
    # SPY benchmark
    benchmark = None
    try:
        spy_df = fetch_prices(["SPY"], start, end)
        if "SPY" in spy_df.columns:
            benchmark = spy_df["SPY"].dropna()
    except Exception:
        pass
    return prices, volume, benchmark


# ─── Ticker Input ─────────────────────────────────────────────────────────────

ticker_input = st.text_input(
    "Ticker", value="SPY", placeholder="e.g. SPY, AAPL, GLD, ^VIX"
).strip().upper()

if not ticker_input:
    st.stop()

prices, volume, benchmark = _fetch_data(ticker_input)

if prices.empty:
    st.error(f"Could not load price data for **{ticker_input}**. Check the ticker and try again.")
    st.stop()

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

divs = _check_divergences(latest)
for msg in divs:
    st.warning(f"Divergence: {msg}")

# Radar chart
radar_vals   = [_normalize_for_radar(latest.get(s, float("nan")), s) for s in RADAR_SIGNALS]
radar_labels = [SIGNAL_LABELS[s] for s in RADAR_SIGNALS]
radar_closed = radar_vals + [radar_vals[0]]
label_closed = radar_labels + [radar_labels[0]]

fig_radar = go.Figure(go.Scatterpolar(
    r=radar_closed,
    theta=label_closed,
    fill="toself",
    fillcolor="rgba(0, 150, 200, 0.15)",
    line=dict(color="rgba(0, 150, 200, 0.8)", width=2),
    hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1],
                        tickvals=[0.25, 0.5, 0.75], ticktext=["", "", ""]),
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
        val   = latest.get(sig, float("nan"))
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

all_signal_keys   = list(SIGNAL_LABELS.keys())
all_signal_names  = list(SIGNAL_LABELS.values())
selected_label    = st.selectbox("Signal", all_signal_names, key="sig_hist")
selected_key      = all_signal_keys[all_signal_names.index(selected_label)]

sig_series = ft[selected_key].dropna()
cutoff_2y  = sig_series.index[-1] - pd.DateOffset(years=2)
sig_2y     = sig_series[sig_series.index >= cutoff_2y]

# Line chart
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=sig_2y.index, y=sig_2y.values,
    mode="lines",
    line=dict(color="#1E90FF", width=1.5),
    name=selected_label,
))
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
cutoff_1y = sig_series.index[-1] - pd.DateOffset(years=1)
sig_1y    = sig_series[sig_series.index >= cutoff_1y]

if not sig_1y.empty:
    dow     = sig_1y.index.dayofweek
    week    = sig_1y.index.isocalendar().week.astype(int)
    col_idx = week - week.min()
    n_weeks = int(col_idx.max()) + 1
    grid    = np.full((7, n_weeks), float("nan"))
    for d, w, v in zip(dow, col_idx, sig_1y.values):
        grid[d, w] = v

    use_rev = selected_key in ("vol_regime_score", "macro_stress")
    fig_cal = go.Figure(go.Heatmap(
        z=grid,
        colorscale="RdYlGn_r" if use_rev else "RdYlGn",
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

for col in ("momentum_short", "momentum_long", "mean_reversion", "relative_strength"):
    if col in display_ft.columns:
        display_ft[col] = display_ft[col].map(
            lambda v: f"{v:.4f}" if not pd.isna(v) else "—"
        )
for col in ("vol_regime_score", "macro_stress", "volume_regime_score", "fiftytwo_week_pos"):
    if col in display_ft.columns:
        display_ft[col] = display_ft[col].map(
            lambda v: f"{v:.1f}" if not pd.isna(v) else "—"
        )
if "trend_strength" in display_ft.columns:
    display_ft["trend_strength"] = display_ft["trend_strength"].map(
        lambda v: f"{v:.3f}" if not pd.isna(v) else "—"
    )

st.dataframe(display_ft, use_container_width=True)
st.caption(f"Showing {len(ft):,} trading days of features.")

today_str = datetime.today().strftime("%Y-%m-%d")
st.download_button(
    label="Download CSV",
    data=ft.to_csv().encode("utf-8"),
    file_name=f"{ticker_input}_features_{today_str}.csv",
    mime="text/csv",
)
