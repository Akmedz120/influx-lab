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
