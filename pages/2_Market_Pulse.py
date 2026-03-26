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
