import streamlit as st

st.set_page_config(page_title="InFlux Lab", layout="wide")

st.title("InFlux Lab")
st.caption("A local research environment for financial modeling, signal development, and market analysis.")

st.divider()

st.markdown("""
This workspace is organized into focused modules. Each module builds on the last.
Use the sidebar to navigate.

| Module | Purpose | Status |
|--------|---------|--------|
| **Foundations** | Returns, distributions, Monte Carlo simulation | ✅ Active |
| **Market Pulse** | Sentiment, fear/greed, risk appetite, macro stress | 🔜 Coming |
| **Integration** | Cross-asset correlations, macro regime analysis | 🔜 Coming |
| **Features** | Signal generation, derived metrics | 🔜 Coming |
| **ML / AI** | Pattern detection, predictive models on structured features | 🔜 Coming |
| **Sectors** | Sector rotation, relative strength, money flow | 🔜 Coming |
| **Sandbox** | Free experimentation | 🔜 Coming |

---
**Start in Foundations** — it's the base everything else is built on.
""")
