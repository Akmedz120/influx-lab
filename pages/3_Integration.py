import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from modules.integration.correlations import get_correlation_matrix
from modules.integration.regimes import (
    get_fear_greed_history, get_regime_history, get_regime_asset_stats,
    REGIME_ORDER, REGIME_COLORS,
)
from modules.integration.leadlag import scan_all_vs_target, cross_correlate
from modules.market_pulse.indicators import (
    get_vix, get_dxy, get_hy_credit_spread,
    get_yield_curve, get_gold_spy_ratio, get_copper_gold_ratio,
    SECTOR_TICKERS, HEATMAP_TICKERS, HEATMAP_LABELS,
)
from modules.data.fetcher import fetch_prices
from modules.ui.glossary import render_definition
from modules.ui.context import so_what, correlation_so_what, regime_so_what, leadlag_so_what

st.set_page_config(page_title="Integration", layout="wide")
st.title("Integration")
st.caption(
    "Cross-asset relationships, macro regime history, and lead-lag analysis. "
    "A hypothesis generator, not a prediction engine."
)


# ─── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_daily_series() -> dict[str, pd.Series]:
    series = {}
    for fn, name in [
        (get_vix,               "VIX"),
        (get_dxy,               "DXY"),
        (get_hy_credit_spread,  "HY Spread"),
        (get_yield_curve,       "Yield Curve"),
        (get_gold_spy_ratio,    "Gold/SPY"),
        (get_copper_gold_ratio, "Copper/Gold"),
    ]:
        try:
            series[name] = fn()["series"]
        except Exception:
            pass

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=1100)).strftime("%Y-%m-%d")
    extra_tickers = list(SECTOR_TICKERS.keys()) + list(HEATMAP_TICKERS)
    try:
        prices = fetch_prices(extra_tickers, start, end)
        for ticker in extra_tickers:
            if ticker in prices.columns:
                label = SECTOR_TICKERS.get(ticker) or HEATMAP_LABELS.get(ticker, ticker)
                series[label] = prices[ticker].dropna()
    except Exception:
        pass

    return series


all_series = _load_daily_series()
all_prices_df = pd.DataFrame(all_series) if all_series else pd.DataFrame()


# ─── Section 1: Cross-Asset Correlations ──────────────────────────────────────

st.header("Cross-Asset Correlations")
st.caption(
    "How do these assets move relative to each other? "
    "Green = move together, red = move opposite. "
    "Try the regime filter to see how correlations shift during stress."
)

try:
    if all_prices_df.empty:
        st.warning("No asset data available.")
    else:
        col_win, col_reg = st.columns([1, 2])
        with col_win:
            window = st.selectbox("Window", [30, 60, 90], index=1, key="corr_window")
        with col_reg:
            regime_filter = st.selectbox(
                "Regime filter",
                ["All periods"] + REGIME_ORDER,
                index=0,
                key="corr_regime",
            )

        regime_dates = None
        if regime_filter != "All periods":
            try:
                fg_hist    = get_fear_greed_history()
                r_hist     = get_regime_history(fg_hist)
                regime_dates = r_hist[r_hist == regime_filter].index
            except Exception:
                st.warning("Could not load regime history for filtering.")

        corr = get_correlation_matrix(all_prices_df, window=window, regime_dates=regime_dates)

        fig_corr = px.imshow(
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_corr.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_colorbar=dict(title="Correlation"),
        )
        fig_corr.update_traces(textfont_size=8)
        st.plotly_chart(fig_corr, use_container_width=True)

        if regime_filter != "All periods":
            n_days = len(regime_dates) if regime_dates is not None else 0
            st.caption(
                f"Showing correlations during **{regime_filter}** periods only ({n_days} trading days)."
            )
        else:
            st.caption(
                "In a crisis, assets that normally move independently tend to converge — "
                "that's when diversification fails. Try filtering by **Extreme Fear** to see this."
            )

        # So What: compute average off-diagonal correlation
        try:
            import numpy as np
            mask = ~np.eye(len(corr), dtype=bool)
            _avg_corr = float(corr.values[mask].mean()) if mask.any() else None
        except Exception:
            _avg_corr = None
        so_what(correlation_so_what(_avg_corr))
        render_definition("correlation")

except Exception as e:
    st.error(f"Could not load correlations: {e}")

st.divider()


# ─── Section 2: Macro Regime ──────────────────────────────────────────────────

st.header("Macro Regime")
st.caption("Market environment classified from the Fear & Greed composite score over 3 years.")

REGIME_DESCRIPTIONS = {
    "Extreme Greed": "Markets are very calm. Risk appetite is elevated, volatility low, credit conditions easy.",
    "Greed":         "Conditions are favorable. Investors are taking on risk, spreads are tight.",
    "Neutral":       "Mixed signals. No dominant fear or greed — watch for a shift in either direction.",
    "Fear":          "Stress is building. Credit spreads widening, volatility rising, defensive rotation starting.",
    "Extreme Fear":  "Fear is dominant. Investors fleeing to safety — gold, bonds, cash.",
}

try:
    fg_history     = get_fear_greed_history()
    regime_history = get_regime_history(fg_history)
    current_regime = regime_history.iloc[-1]
    current_color  = REGIME_COLORS[current_regime]

    st.markdown(
        f"<div style='padding:16px;background:{current_color}20;"
        f"border-left:4px solid {current_color};border-radius:4px'>"
        f"<b style='font-size:1.1em'>Current Regime: "
        f"<span style='color:{current_color}'>{current_regime}</span></b><br>"
        f"{REGIME_DESCRIPTIONS.get(current_regime, '')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Regime timeline
    fig_tl = go.Figure()
    for regime in REGIME_ORDER:
        mask   = regime_history == regime
        dates  = fg_history[mask].index
        scores = fg_history[mask].values
        fig_tl.add_trace(go.Scatter(
            x=dates, y=scores,
            mode="markers",
            marker=dict(color=REGIME_COLORS[regime], size=4),
            name=regime,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} (" + regime + ")<extra></extra>",
        ))
    for i, (y0, y1) in enumerate([(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]):
        regime_for_band = REGIME_ORDER[i]
        fig_tl.add_hrect(
            y0=y0, y1=y1,
            fillcolor=REGIME_COLORS[regime_for_band],
            opacity=0.05, line_width=0,
        )
    fig_tl.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="F&G Score", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # Per-regime asset stats
    st.markdown("**Average daily return by regime** (key macro indicators)")
    STAT_ASSETS = ["VIX", "DXY", "HY Spread", "Yield Curve", "Gold/SPY", "Copper/Gold"]
    stat_df_input = all_prices_df[[c for c in STAT_ASSETS if c in all_prices_df.columns]]
    if not stat_df_input.empty:
        stats = get_regime_asset_stats(regime_history, stat_df_input)
        fmt = stats.apply(
            lambda col: col.map(lambda v: f"{v:.3%}" if not pd.isna(v) else "—")
        )
        st.dataframe(fmt, use_container_width=True)
    else:
        st.caption("No asset data available for regime stats.")

    so_what(regime_so_what(current_regime), color=current_color)
    render_definition("macro_regime")

except Exception as e:
    st.error(f"Could not load regime data: {e}")

st.divider()


# ─── Section 3: Lead-Lag Scanner ──────────────────────────────────────────────

st.header("Lead-Lag Scanner")
st.caption(
    "Which indicators move *before* others? "
    "Pick a target, click **Scan** — we rank all indicators by how strongly they "
    "lead or lag the target over the past 3 years."
)
so_what(leadlag_so_what())
render_definition("lead_lag")

TARGET_OPTIONS = ["VIX", "DXY", "HY Spread", "Yield Curve", "Gold/SPY", "Copper/Gold", "S&P 500 (US)"]

try:
    available_targets = [t for t in TARGET_OPTIONS if t in all_series]
    if not available_targets:
        st.warning("No target series available.")
    else:
        col_t, col_btn = st.columns([3, 1])
        with col_t:
            selected_target = st.selectbox("Target", available_targets, key="ll_target")
        with col_btn:
            st.write("")
            run_scan = st.button("Scan", key="ll_scan", type="primary")

        if run_scan:
            target_s  = all_series[selected_target].rename(selected_target)
            scan_pool = {k: v for k, v in all_series.items() if k != selected_target}
            with st.spinner(f"Scanning {len(scan_pool)} indicators vs {selected_target}..."):
                results = scan_all_vs_target(target_s, scan_pool, max_lag=60)
            st.session_state["ll_results"]     = results
            st.session_state["ll_target_name"] = selected_target

        if "ll_results" in st.session_state:
            results     = st.session_state["ll_results"]
            target_name = st.session_state["ll_target_name"]

            if not results.empty:
                st.markdown(f"**Ranked relationships with {target_name}**")
                display = results[["indicator_name", "peak_correlation", "peak_lag", "interpretation"]].copy()
                display["peak_correlation"] = display["peak_correlation"].map(lambda v: f"{v:+.3f}")
                st.dataframe(display, use_container_width=True, hide_index=True)

                st.markdown("**Lag profile for selected indicator**")
                detail = st.selectbox(
                    "Inspect indicator",
                    results["indicator_name"].tolist(),
                    key="ll_detail",
                )
                if detail and detail in all_series:
                    cc = cross_correlate(
                        all_series[detail],
                        all_series[target_name].rename(target_name),
                        max_lag=60,
                    )
                    peak_lag = int(cc.abs().idxmax())

                    fig_cc = go.Figure(go.Bar(
                        x=cc.index.tolist(),
                        y=cc.values,
                        marker_color=["#00CC96" if v >= 0 else "#EF553B" for v in cc.values],
                    ))
                    fig_cc.add_vline(
                        x=0, line_dash="dash", line_color="gray", line_width=1,
                    )
                    fig_cc.add_vline(
                        x=peak_lag, line_dash="dot", line_color="#FFA500", line_width=2,
                        annotation_text=f"peak: lag {peak_lag}",
                        annotation_position="top right",
                    )
                    fig_cc.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis_title="Lag (trading days)",
                        yaxis=dict(title="Correlation", range=[-1, 1]),
                    )
                    st.plotly_chart(fig_cc, use_container_width=True)
                    st.caption(
                        "Bars to the **right** (positive lag) = indicator moves *before* the target. "
                        "Bars to the **left** (negative lag) = indicator *follows* the target. "
                        "Orange line = peak correlation."
                    )

except Exception as e:
    st.error(f"Could not load lead-lag scanner: {e}")
