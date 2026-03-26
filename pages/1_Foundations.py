import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.data.fetcher import fetch_prices
from modules.foundations.returns import calculate_returns, compute_stats
from modules.foundations.distributions import fit_normal, normal_pdf_range, normality_test
from modules.foundations.monte_carlo import gbm_simulation, simulation_stats

st.set_page_config(page_title="Foundations", layout="wide")
st.title("Foundations")
st.caption("Understand return distributions and uncertainty — the base everything else builds on.")

# ─── Section 1: Returns & Distributions ──────────────────────────────────────
st.header("Returns & Distributions")

col1, col2 = st.columns([3, 1])
with col1:
    tickers_input = st.text_input("Tickers (comma separated)", value="SPY, QQQ")
with col2:
    period = st.selectbox("Lookback Period", ["1Y", "2Y", "5Y", "10Y"], index=1)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
period_days = {"1Y": 365, "2Y": 730, "5Y": 1825, "10Y": 3650}
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=period_days[period])).strftime("%Y-%m-%d")

if tickers:
    with st.spinner("Loading data..."):
        try:
            prices = fetch_prices(tickers, start_date, end_date)
            returns_dict = {t: calculate_returns(prices[t]) for t in tickers if t in prices.columns}
            stats_dict = {t: compute_stats(r) for t, r in returns_dict.items()}

            # Distribution chart
            fig = go.Figure()
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
            for i, ticker in enumerate(returns_dict):
                r = returns_dict[ticker]
                color = colors[i % len(colors)]
                fig.add_trace(go.Histogram(
                    x=r, name=ticker, opacity=0.55,
                    histnorm="probability density", nbinsx=80,
                    marker_color=color
                ))
                x_norm, y_norm = normal_pdf_range(r)
                fig.add_trace(go.Scatter(
                    x=x_norm, y=y_norm,
                    name=f"{ticker} — normal fit",
                    line=dict(color=color, dash="dash", width=2)
                ))
            fig.update_layout(
                title="Return Distributions vs Normal",
                xaxis_title="Daily Return",
                yaxis_title="Probability Density",
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats table
            stats_df = pd.DataFrame(stats_dict).T
            stats_df.index.name = "Ticker"
            display_df = pd.DataFrame({
                "Ann. Return": stats_df["mean_return"].map("{:.1%}".format),
                "Ann. Volatility": stats_df["volatility"].map("{:.1%}".format),
                "Skewness": stats_df["skewness"].map("{:.2f}".format),
                "Excess Kurtosis": stats_df["kurtosis"].map("{:.2f}".format),
                "Sharpe (no Rf)": stats_df["sharpe"].map("{:.2f}".format),
            })
            st.dataframe(display_df, use_container_width=True)

            # So what — one insight per ticker
            for ticker, s in stats_dict.items():
                nt = normality_test(returns_dict[ticker])
                skew_desc = (
                    "negatively skewed — losses tend to be larger than gains"
                    if s["skewness"] < -0.3
                    else "positively skewed — gains tend to be larger than losses"
                    if s["skewness"] > 0.3
                    else "roughly symmetric"
                )
                kurt_desc = (
                    "fat tails — extreme moves happen more often than a normal model would predict"
                    if s["kurtosis"] > 1
                    else "thin tails — fewer extreme moves than a normal model"
                    if s["kurtosis"] < -0.5
                    else "near-normal tail behavior"
                )
                normal_note = "" if nt["is_normal"] else " The returns are **not** normally distributed (Shapiro-Wilk p < 0.05) — standard risk models may understate actual risk."
                st.info(
                    f"**{ticker}**: Returns are {skew_desc}, with {kurt_desc}. "
                    f"Annualized volatility: {s['volatility']:.1%}, Sharpe: {s['sharpe']:.2f}.{normal_note}"
                )

        except Exception as e:
            st.error(f"Could not load data: {e}")

# ─── Section 2: Monte Carlo ───────────────────────────────────────────────────
st.divider()
st.header("Monte Carlo Simulation")
st.caption(
    "Simulates 1,000 possible future price paths using Geometric Brownian Motion (GBM). "
    "GBM assumes log-normal returns — it **underestimates tail risk**. "
    "This is a range of outcomes, not a forecast."
)

mc_col1, mc_col2, mc_col3 = st.columns(3)
with mc_col1:
    mc_ticker = st.text_input("Ticker", value="SPY", key="mc_ticker")
with mc_col2:
    horizon_label = st.selectbox("Horizon", ["30 days", "90 days", "1 year"], index=1)
with mc_col3:
    n_sims = st.selectbox("Paths", [500, 1000, 2000], index=1)

horizon_map = {"30 days": 30, "90 days": 90, "1 year": 252}
T = horizon_map[horizon_label]

if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        try:
            mc_prices = fetch_prices([mc_ticker], start_date, end_date)
            mc_col = mc_ticker if mc_ticker in mc_prices.columns else mc_prices.columns[0]
            mc_rets = calculate_returns(mc_prices[mc_col])
            mu_d = float(mc_rets.mean())
            sigma_d = float(mc_rets.std())
            S0 = float(mc_prices[mc_col].iloc[-1])

            paths = gbm_simulation(S0=S0, mu=mu_d, sigma=sigma_d, T=T, n_simulations=n_sims)
            mc_stats = simulation_stats(paths)

            # Fan chart — subsample to cap at ~100 visible paths for performance
            fig2 = go.Figure()
            x_axis = list(range(T + 1))
            step = max(1, n_sims // 100)
            for i in range(0, n_sims, step):
                fig2.add_trace(go.Scatter(
                    y=paths[:, i], x=x_axis, mode="lines",
                    line=dict(color="rgba(100,149,237,0.15)", width=1),
                    showlegend=False, hoverinfo="skip"
                ))
            # Percentile lines
            fig2.add_trace(go.Scatter(y=np.percentile(paths, 5, axis=1), x=x_axis, name="5th pct (bad)", line=dict(color="#EF553B", width=2.5)))
            fig2.add_trace(go.Scatter(y=np.percentile(paths, 50, axis=1), x=x_axis, name="Median", line=dict(color="#00CC96", width=2.5)))
            fig2.add_trace(go.Scatter(y=np.percentile(paths, 95, axis=1), x=x_axis, name="95th pct (good)", line=dict(color="#636EFA", width=2.5)))

            fig2.update_layout(
                title=f"{mc_ticker} — {n_sims} simulated paths over {horizon_label}",
                xaxis_title="Trading Days",
                yaxis_title="Price ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Starting Price", f"${S0:.2f}")
            m2.metric("Median Outcome", f"${mc_stats['median']:.2f}", delta=f"{(mc_stats['median']/S0 - 1):.1%}")
            m3.metric("Worst 5% Scenario", f"${mc_stats['p5']:.2f}", delta=f"{(mc_stats['p5']/S0 - 1):.1%}", delta_color="inverse")
            m4.metric("Probability of Loss", f"{mc_stats['prob_loss']:.1%}")

            st.info(
                f"**So what:** Starting at ${S0:.2f}, the median path ends at ${mc_stats['median']:.2f} after {horizon_label}. "
                f"In the worst 5% of scenarios, the price falls to ${mc_stats['p5']:.2f} ({(mc_stats['p5']/S0 - 1):.1%}). "
                f"There is a {mc_stats['prob_loss']:.1%} probability of loss based on historical return patterns. "
                f"The wide spread in paths reflects genuine uncertainty — not imprecision in the model."
            )

        except Exception as e:
            st.error(f"Could not run simulation: {e}")
