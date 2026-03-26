"""
Glossary of financial/market terms with plain-English definitions.
Renders as collapsible st.expander blocks anywhere in the app.
"""
import streamlit as st

GLOSSARY: dict[str, dict] = {
    "vix": {
        "name": "VIX",
        "plain_name": "The Market's Fear Gauge",
        "what": "A number that measures how much uncertainty and fear exists in the stock market right now.",
        "why": "When traders are scared, they pay more for insurance on their investments — that drives the VIX up.",
        "example": "VIX of 12 → calm, quiet market. VIX of 40 → panic, like during a crash.",
        "high": "Markets are fearful. Volatility is spiking. Could be a buying opportunity — or things get worse first.",
        "low": "Markets are calm and complacent. Historically, very low VIX can precede a surprise shock.",
    },
    "yield_curve": {
        "name": "Yield Curve (10Y − 2Y)",
        "plain_name": "Are Long-Term Bonds Paying More Than Short-Term?",
        "what": "The difference between 10-year and 2-year US Treasury yields. Normally positive (long pays more).",
        "why": "When short-term rates are higher than long-term, it means the market expects future rates to fall — often because a recession is coming.",
        "example": "Spread of +1.5% = normal. Spread of −0.5% = inverted — historically precedes recessions by 12–18 months.",
        "high": "Positive spread: normal growth expectations. Economy looks healthy.",
        "low": "Inverted (negative): markets expect a slowdown. One of the most reliable recession signals historically.",
    },
    "hy_spread": {
        "name": "HY Credit Spread",
        "plain_name": "How Much Extra Return Do Risky Bonds Demand?",
        "what": "The extra interest rate that junk bonds (risky company debt) pay vs safe US Treasuries.",
        "why": "When investors get nervous, they demand more compensation for taking credit risk — spreads widen.",
        "example": "2% spread = calm, lenders are comfortable. 8% spread = fear, like during the 2020 COVID crash.",
        "high": "Spreads are wide. Credit stress — companies may struggle to borrow. Risk-off environment.",
        "low": "Spreads are tight. Easy credit conditions, markets are comfortable extending risk.",
    },
    "dxy": {
        "name": "DXY (Dollar Index)",
        "plain_name": "How Strong Is the US Dollar?",
        "what": "A measure of the US dollar's value against a basket of major currencies (euro, yen, pound, etc.).",
        "why": "A strong dollar hurts emerging markets and US exports. In crises, the dollar usually surges as a safe haven.",
        "example": "DXY 90 = relatively weak dollar. DXY 105 = strong dollar.",
        "high": "Dollar is strong. Can pressure commodities, EM assets, and US multinational earnings.",
        "low": "Dollar is weak. Often good for gold, commodities, and international stocks.",
    },
    "fear_greed": {
        "name": "Fear & Greed Index",
        "plain_name": "What's the Overall Market Mood?",
        "what": "A composite score (0–100) that blends multiple market signals into a single mood reading.",
        "why": "Markets swing between fear (everyone selling) and greed (everyone buying). This index tracks where we are.",
        "example": "Score 10 = extreme fear, possibly a buying opportunity. Score 90 = extreme greed, possibly overheated.",
        "high": "Extreme Greed (>75): Markets are frothy. History says to be cautious.",
        "low": "Extreme Fear (<25): Panic is high. Historically, these are the best long-term entry points.",
    },
    "correlation": {
        "name": "Correlation",
        "plain_name": "Do Two Things Move Together?",
        "what": "A number from −1 to +1 showing how closely two assets move together.",
        "why": "For diversification to work, you want assets that don't all fall at the same time.",
        "example": "+0.9 = almost always move together. 0 = no relationship. −0.9 = usually move in opposite directions.",
        "high": "High positive correlation: assets move in sync. Less diversification benefit.",
        "low": "Negative correlation: one tends to rise when the other falls. Good for hedging.",
    },
    "macro_regime": {
        "name": "Macro Regime",
        "plain_name": "What Phase Is the Market In?",
        "what": "A classification of the current market environment (Extreme Greed → Extreme Fear).",
        "why": "The same asset can behave very differently depending on whether we're in a risk-on or risk-off environment.",
        "example": "During Extreme Fear: cash and gold tend to outperform. During Extreme Greed: growth stocks tend to lead.",
        "high": "Extreme Fear: defensive assets typically outperform.",
        "low": "Extreme Greed: risk assets typically outperform.",
    },
    "lead_lag": {
        "name": "Lead-Lag Relationship",
        "plain_name": "Does One Indicator Move Before Another?",
        "what": "When one market signal tends to move a few days or weeks BEFORE another.",
        "why": "If you can identify what leads what, you get an early warning system.",
        "example": "HY spreads often widen before VIX spikes — spreads can lead by 5–15 days.",
        "high": "Strong positive lead-lag: reliable early warning. Worth watching when context changes.",
        "low": "Weak lead-lag: the relationship is noisy and may not be actionable.",
    },
    "volatility_regime": {
        "name": "Volatility Regime",
        "plain_name": "Is This Asset Choppier Than Usual?",
        "what": "A 0–100 score showing how the asset's recent volatility compares to its own 1-year history.",
        "why": "Context matters — a VIX of 20 means something different depending on whether it's been averaging 12 or 30.",
        "example": "Score 90 = unusually volatile right now. Score 10 = unusually calm.",
        "high": "Volatility is elevated vs history. Options are expensive, moves can be large.",
        "low": "Volatility is suppressed. Calm periods can end suddenly — low vol often precedes spikes.",
    },
    "momentum": {
        "name": "Momentum",
        "plain_name": "Is the Price Going Up or Down?",
        "what": "How much the price has moved over a recent period (20 or 90 days), expressed as a percentage.",
        "why": "Trending assets tend to keep trending — momentum is one of the most robust factors in finance.",
        "example": "+8% over 20 days = strong uptrend. −12% = downtrend.",
        "high": "Positive: price has been rising. Trend is your friend — until it isn't.",
        "low": "Negative: price has been falling. Caution — could be a buying opportunity or a falling knife.",
    },
    "mean_reversion": {
        "name": "Mean Reversion (Z-Score)",
        "plain_name": "Is This Price Stretched From Its Average?",
        "what": "How many standard deviations the current price is above or below its 60-day average.",
        "why": "Prices tend to drift back toward their average over time — knowing how stretched we are helps gauge risk.",
        "example": "Z-score +2 = price is 2 standard deviations above average, quite stretched. −1.5 = below average.",
        "high": "Large positive z-score: price is stretched upward, elevated pullback risk.",
        "low": "Large negative z-score: price is stretched downward, may be oversold.",
    },
    "trend_strength": {
        "name": "Trend Strength (R²)",
        "plain_name": "How Linear and Consistent Is the Trend?",
        "what": "An R² value (0–1) from a regression of the price over the past 60 days. Near 1 = very clean trend.",
        "why": "Not all trends are equal — a jagged price going up has weaker trend strength than a smooth one.",
        "example": "R² 0.95 = near-perfect linear trend. R² 0.2 = choppy, no clear direction.",
        "high": "Strong consistent trend — reliable direction to follow.",
        "low": "Weak or absent trend — noisy price action, signals less reliable.",
    },
    "relative_strength": {
        "name": "Relative Strength vs SPY",
        "plain_name": "Is This Asset Beating the Market?",
        "what": "The asset's 20-day return minus SPY's 20-day return. Positive = outperforming the S&P 500.",
        "why": "In a risk-on environment, outperformers tend to keep outperforming. In a risk-off, they may hold up better.",
        "example": "+5% relative strength: asset beat SPY by 5% over 20 days.",
        "high": "Outperforming. Could be leadership — or mean-revert if stretched too far.",
        "low": "Underperforming. Lagging the market — could be sector rotation or a red flag.",
    },
    "fiftytwo_week_pos": {
        "name": "52-Week Position",
        "plain_name": "Where Is This Price in Its Year Range?",
        "what": "Where the current price sits in its 52-week high-to-low range (0% = year low, 100% = year high).",
        "why": "Gives quick context on whether this is near a breakdown or a breakout point.",
        "example": "95% = near 52-week high. 5% = near 52-week low.",
        "high": "Near 52-week high: strong trend, but watch for resistance.",
        "low": "Near 52-week low: potential value — or further downside if trend continues.",
    },
    "volume_regime": {
        "name": "Volume Regime",
        "plain_name": "Is Trading Volume High or Low vs Normal?",
        "what": "A 0–100 score comparing current trading volume to the past year's volume history.",
        "why": "Price moves on high volume are more meaningful than moves on thin volume.",
        "example": "Score 85 = unusually high volume. Score 10 = unusually low volume.",
        "high": "Elevated volume. Confirms price moves — breakouts and breakdowns more reliable.",
        "low": "Low volume. Price moves may be noise. Watch for a high-volume candle to confirm direction.",
    },
    "sector_rotation": {
        "name": "Sector Rotation",
        "plain_name": "What Part of the Economy Is Leading?",
        "what": "Tracking which market sectors are gaining and losing. Risk-On (tech, cons. disc.) vs Risk-Off (utilities, healthcare).",
        "why": "Sector leadership tells you whether investors are taking risk or hiding in defensive names.",
        "example": "Tech and consumer discretionary leading = risk-on. Utilities and healthcare leading = defensive rotation.",
        "high": "Risk-On sectors leading: growth appetite, bull market behavior.",
        "low": "Risk-Off sectors leading: defensive positioning, late-cycle or stressed environment.",
    },
    "macro_stress": {
        "name": "Macro Stress Score",
        "plain_name": "How Stressed Is the Overall Market Environment?",
        "what": "The Fear & Greed score (0 = calm/greed, 100 = extreme fear/stress), reindexed to match this asset's trading calendar.",
        "why": "Asset signals look different in a stressed vs calm macro backdrop. This provides that backdrop.",
        "example": "Score 20 = calm, greed dominant. Score 80 = fear, markets under stress.",
        "high": "High stress (>67): macro backdrop is fearful. Combine with asset signals carefully.",
        "low": "Low stress (<33): calm macro. Asset-specific signals dominate over macro noise.",
    },
}


def render_definition(term_key: str) -> None:
    """Render a collapsible definition expander for a given GLOSSARY term key."""
    if term_key not in GLOSSARY:
        return
    g = GLOSSARY[term_key]
    with st.expander(f"What is {g['name']}?"):
        st.markdown(f"**{g['plain_name']}**")
        st.markdown(g["what"])
        st.markdown(f"*Why it matters:* {g['why']}")
        st.markdown(f"*Example:* {g['example']}")
        col_h, col_l = st.columns(2)
        with col_h:
            st.markdown(f"**When high:** {g['high']}")
        with col_l:
            st.markdown(f"**When low:** {g['low']}")
