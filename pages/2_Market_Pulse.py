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
from modules.ui.glossary import render_definition
from modules.ui.context import (
    so_what, fg_so_what, sentiment_so_what,
    risk_appetite_so_what, macro_so_what,
    liquidity_so_what, sector_so_what,
)

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

    so_what(fg_so_what(fg["score"], fg["label"]), color=gauge_color)
    render_definition("fear_greed")

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

_sent_colors = {}
c1, c2, c3 = st.columns(3)
for col, fn, key in [(c1, get_vix, "vix"), (c2, get_aaii_sentiment, "aaii"), (c3, get_consumer_confidence, "conf")]:
    try:
        ind = fn()
        sc  = _score(ind)
        _sent_colors[key] = sc["color"]
        _indicator_block(col, ind, sc)
    except Exception as e:
        _sent_colors[key] = "yellow"
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

so_what(
    sentiment_so_what(
        _sent_colors.get("vix", "yellow"),
        _sent_colors.get("aaii", "yellow"),
        _sent_colors.get("conf", "yellow"),
    )
)
render_definition("vix")

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

_ra_colors = {}
c4, c5, c6, c7 = st.columns(4)
for col, fn, key in [
    (c4, get_hy_credit_spread,  "hy"),
    (c5, get_gold_spy_ratio,    "gold"),
    (c6, get_dxy,               "dxy"),
    (c7, get_copper_gold_ratio, "cg"),
]:
    try:
        ind = fn()
        sc  = _score(ind)
        _ra_colors[key] = sc["color"]
        _indicator_block(col, ind, sc)
    except Exception as e:
        _ra_colors[key] = "yellow"
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

so_what(
    risk_appetite_so_what(
        _ra_colors.get("hy", "yellow"),
        _ra_colors.get("gold", "yellow"),
        _ra_colors.get("dxy", "yellow"),
        _ra_colors.get("cg", "yellow"),
    )
)
render_definition("hy_spread")
render_definition("dxy")

st.divider()

# ─── Macro Stress ─────────────────────────────────────────────────────────────
st.header("Macro Stress")

_yc_val = None
_fed_color = "yellow"
c8, c9 = st.columns(2)
for col, fn in [(c8, get_yield_curve), (c9, get_fed_funds)]:
    try:
        ind = fn()
        sc  = _score(ind)
        note = None
        if "Yield Curve" in ind["label"]:
            _yc_val = ind["current"]
            if ind["current"] < 0:
                note = f"Curve is **inverted** ({ind['current']:.2f}%). Historically precedes recession by 12–18 months."
            elif ind["current"] < 0.5:
                note = f"Curve is near-flat ({ind['current']:.2f}%). Watch for inversion."
            else:
                note = f"Curve is positive ({ind['current']:.2f}%). Normal — longer rates above short-term."
        else:
            _fed_color = sc["color"]
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

so_what(macro_so_what(_yc_val, _fed_color))
render_definition("yield_curve")

st.divider()

# ─── Liquidity ────────────────────────────────────────────────────────────────
st.header("Liquidity")

_liq_colors = {}
c10, c11 = st.columns(2)
for col, fn, key in [(c10, get_m2, "m2"), (c11, get_fed_balance_sheet, "fed")]:
    try:
        ind = fn()
        sc  = _score(ind)
        _liq_colors[key] = sc["color"]
        _indicator_block(col, ind, sc)
    except Exception as e:
        _liq_colors[key] = "yellow"
        with col:
            st.metric(label="—", value="—")
            st.caption(f"Could not load: {e}")

so_what(liquidity_so_what(_liq_colors.get("m2", "yellow"), _liq_colors.get("fed", "yellow")))

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
        ron  = sec_df[sec_df["risk_type"] == "Risk-On"].copy()
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

        _ron_avg  = ron["1w"].mean()  if "1w" in ron.columns  else None
        _roff_avg = roff["1w"].mean() if "1w" in roff.columns else None
        so_what(sector_so_what(_ron_avg, _roff_avg))
        render_definition("sector_rotation")
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
