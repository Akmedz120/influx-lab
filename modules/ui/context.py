"""
Rule-based plain-English "So What" callout boxes for each page section.
Each function returns a string that's rendered inside a colored callout.
"""
import streamlit as st
import pandas as pd


# ─── Core renderer ─────────────────────────────────────────────────────────────

def so_what(text: str, color: str = "#1E90FF") -> None:
    """Render a colored callout box with plain-English context."""
    st.markdown(
        f"<div style='padding:12px 16px;margin:8px 0 4px 0;"
        f"background:{color}18;border-left:4px solid {color};"
        f"border-radius:4px;font-size:0.93em'>"
        f"<b style='color:{color}'>What does this mean?</b><br>{text}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─── Market Pulse contexts ──────────────────────────────────────────────────────

def fg_so_what(score: float, label: str) -> str:
    if label in ("Extreme Fear", "Fear"):
        return (
            f"The market is in <b>{label}</b> (score {score:.0f}/100). "
            "Historically, high fear readings are associated with better forward returns — "
            "but they can stay elevated for weeks. "
            "This doesn't mean 'buy everything now,' it means be aware that pessimism is priced in."
        )
    elif label == "Neutral":
        return (
            f"The market is <b>Neutral</b> (score {score:.0f}/100). "
            "No strong signal in either direction. "
            "Watch the components below to see what's pushing toward stress or calm."
        )
    else:  # Greed / Extreme Greed
        return (
            f"The market is in <b>{label}</b> (score {score:.0f}/100). "
            "Investors are comfortable taking risk right now. "
            "High greed can persist, but the higher the score, the less cushion if sentiment shifts."
        )


def sentiment_so_what(vix_color: str, aaii_color: str, conf_color: str) -> str:
    reds = sum(1 for c in [vix_color, aaii_color, conf_color] if c == "red")
    greens = sum(1 for c in [vix_color, aaii_color, conf_color] if c == "green")
    if reds >= 2:
        return (
            "Multiple sentiment indicators are stressed. "
            "Retail investors are bearish, consumer confidence is weak, and/or fear is elevated. "
            "This kind of broad pessimism often marks bottoms — but can also precede more pain."
        )
    elif greens >= 2:
        return (
            "Sentiment is broadly positive across indicators. "
            "Investors feel good, volatility is low, confidence is up. "
            "A calm environment — but complacency can set in at these levels."
        )
    else:
        return (
            "Sentiment signals are mixed. Some indicators show stress, others show calm. "
            "No strong consensus — worth checking back after a directional move in the market."
        )


def risk_appetite_so_what(hy_color: str, gold_color: str, dxy_color: str, cg_color: str) -> str:
    risk_off = sum(1 for c in [hy_color, gold_color] if c == "red") + \
               sum(1 for c in [dxy_color, cg_color] if c == "red")
    risk_on = sum(1 for c in [hy_color, gold_color] if c == "green") + \
              sum(1 for c in [cg_color] if c == "green")
    if risk_off >= 3:
        return (
            "Risk appetite is contracting. Credit spreads are wide, gold is in demand, "
            "and the copper/gold ratio is falling. Investors are moving to safety. "
            "This is a defensive environment — equities may face headwinds."
        )
    elif risk_on >= 2:
        return (
            "Risk appetite is healthy. Credit spreads are tight and industrial metals are outpacing gold. "
            "Investors are comfortable taking on risk. This is typically a tailwind for equities and high-yield."
        )
    else:
        return (
            "Risk appetite signals are mixed. Some indicators point to caution, others look fine. "
            "No clear risk-on or risk-off tilt right now."
        )


def macro_so_what(yield_curve_val: float | None, fed_color: str) -> str:
    if yield_curve_val is None:
        return "Macro data unavailable. Check back when yield curve data loads."
    if yield_curve_val < 0:
        return (
            f"The yield curve is <b>inverted</b> ({yield_curve_val:.2f}%). "
            "This is one of the most historically reliable recession signals — "
            "recessions have followed inversions by 12–18 months on average. "
            "It doesn't mean a recession is certain or imminent, but it warrants caution on long-term positioning."
        )
    elif yield_curve_val < 0.5:
        return (
            f"The yield curve is <b>near-flat</b> ({yield_curve_val:.2f}%). "
            "It's not inverted yet, but worth watching. "
            "A flat curve means short and long rates are converging — often a sign of economic uncertainty ahead."
        )
    else:
        return (
            f"The yield curve is <b>positively sloped</b> ({yield_curve_val:.2f}%). "
            "Normal macro conditions — longer-term rates are higher, reflecting growth expectations. "
            "No recession signal here."
        )


def liquidity_so_what(m2_color: str, fed_color: str) -> str:
    if m2_color == "green" and fed_color == "green":
        return (
            "Liquidity conditions are easy. The Fed's balance sheet and money supply are expanding. "
            "Historically, loose liquidity is a tailwind for risk assets."
        )
    elif m2_color == "red" or fed_color == "red":
        return (
            "Liquidity is tightening. The Fed is reducing its balance sheet and/or money supply growth is slowing. "
            "Tighter liquidity typically pressures valuations — especially for high-multiple growth stocks."
        )
    else:
        return (
            "Liquidity signals are neutral. Neither strongly expansionary nor contractionary. "
            "Watch for a shift in Fed language or M2 trend."
        )


def sector_so_what(ron_avg: float | None, roff_avg: float | None) -> str:
    if ron_avg is None or roff_avg is None:
        return "Sector data unavailable."
    diff = ron_avg - roff_avg
    if diff > 0.005:
        return (
            f"Risk-On sectors (tech, consumer) are outpacing Risk-Off (utilities, healthcare) "
            f"by about {diff:.1%} recently. "
            "Investors are reaching for growth — broadly constructive for the market."
        )
    elif diff < -0.005:
        return (
            f"Risk-Off sectors (utilities, healthcare) are outpacing Risk-On by about {abs(diff):.1%}. "
            "Investors are rotating into defensives — a sign of caution or late-cycle behavior."
        )
    else:
        return (
            "Risk-On and Risk-Off sectors are performing similarly. "
            "No clear rotation signal — the market is broadly range-bound or mixed."
        )


# ─── Integration page contexts ─────────────────────────────────────────────────

def correlation_so_what(avg_off_diagonal: float | None) -> str:
    if avg_off_diagonal is None:
        return "Correlation data unavailable."
    if avg_off_diagonal > 0.5:
        return (
            f"Average cross-asset correlation is high ({avg_off_diagonal:.2f}). "
            "When assets all move together like this, diversification breaks down — "
            "a portfolio of 10 assets may behave like 2 or 3 when they're all correlated. "
            "This is common during stress episodes."
        )
    elif avg_off_diagonal < 0.2:
        return (
            f"Average cross-asset correlation is low ({avg_off_diagonal:.2f}). "
            "Assets are moving more independently — diversification is working as expected. "
            "This is a healthier environment for multi-asset portfolios."
        )
    else:
        return (
            f"Cross-asset correlation is moderate ({avg_off_diagonal:.2f}). "
            "Some co-movement, but not extreme. "
            "Watch how this shifts if market stress increases — correlations tend to spike in downturns."
        )


def regime_so_what(current_regime: str) -> str:
    descriptions = {
        "Extreme Greed": (
            "We're in <b>Extreme Greed</b>. Markets are at their most complacent — "
            "volatility is low, credit is easy, risk appetite is elevated. "
            "Historically, this is when investors are most exposed to a surprise shock. "
            "Doesn't mean sell everything — but be aware of how much risk you're carrying."
        ),
        "Greed": (
            "We're in a <b>Greed</b> regime. Conditions are favorable, spreads are tight, "
            "investors are taking risk. A broadly supportive environment for equities. "
            "Watch for signs of overheating."
        ),
        "Neutral": (
            "We're in a <b>Neutral</b> regime. No strong fear or greed. "
            "Asset behavior is more idiosyncratic — individual stock or sector factors matter more than macro. "
            "A direction change could go either way."
        ),
        "Fear": (
            "We're in a <b>Fear</b> regime. Stress is building — spreads widening, "
            "volatility rising, rotation into defensives. "
            "Assets with high beta to risk (tech, HY) typically underperform in this environment."
        ),
        "Extreme Fear": (
            "We're in <b>Extreme Fear</b>. Full risk-off mode. "
            "Cash, gold, and treasuries tend to outperform. "
            "Historically, sustained extreme fear levels have preceded the best long-term entry points — "
            "but the bottom is impossible to time precisely."
        ),
    }
    return descriptions.get(current_regime, f"Current regime: {current_regime}.")


def leadlag_so_what() -> str:
    return (
        "This table shows which indicators <b>move before</b> others. "
        "A positive lag means the indicator tends to change direction before the target does — "
        "giving you an early heads-up. "
        "A high correlation at a non-zero lag is more interesting than at lag 0 (which is just regular correlation). "
        "Use this to build a watch list: if HY spreads lead VIX by 10 days, "
        "and spreads are rising today, VIX may follow."
    )


# ─── Features page contexts ────────────────────────────────────────────────────

def feature_so_what(latest: "pd.Series") -> str:  # type: ignore[name-defined]
    """
    Generate a plain-English summary of the current feature state.
    Uses the latest row from the feature table.
    """
    lines = []

    # Volatility
    vol = latest.get("vol_regime_score")
    vol_label = latest.get("vol_regime_label", "mid")
    if pd.notna(vol):
        if vol_label == "high":
            lines.append(f"Volatility is <b>elevated</b> (score {vol:.0f}/100) — the asset is choppier than usual.")
        elif vol_label == "low":
            lines.append(f"Volatility is <b>suppressed</b> (score {vol:.0f}/100) — unusually calm. Can change fast.")
        else:
            lines.append(f"Volatility is in a <b>normal range</b> (score {vol:.0f}/100).")

    # Momentum
    ms = latest.get("momentum_short")
    ml = latest.get("momentum_long")
    if pd.notna(ms) and pd.notna(ml):
        if ms > 0 and ml > 0:
            lines.append("Both short- and long-term momentum are <b>positive</b> — price has been trending up.")
        elif ms < 0 and ml < 0:
            lines.append("Both short- and long-term momentum are <b>negative</b> — price has been in a downtrend.")
        elif ms > 0 and ml < 0:
            lines.append("Short-term momentum is up but the longer trend is still down — a potential <b>early reversal</b> to watch.")
        elif ms < 0 and ml > 0:
            lines.append("Recent price action is weakening even though the long trend is positive — <b>momentum fading</b>.")

    # Mean reversion
    mr = latest.get("mean_reversion")
    if pd.notna(mr):
        if mr > 2:
            lines.append(f"Price is <b>well above</b> its 60-day average (z-score {mr:.1f}) — stretched upward, elevated pullback risk.")
        elif mr < -2:
            lines.append(f"Price is <b>well below</b> its 60-day average (z-score {mr:.1f}) — potentially oversold.")

    # 52-week position
    pos = latest.get("fiftytwo_week_pos")
    if pd.notna(pos):
        if pos > 90:
            lines.append(f"Price is near its <b>52-week high</b> ({pos:.0f}%) — in a strong uptrend.")
        elif pos < 10:
            lines.append(f"Price is near its <b>52-week low</b> ({pos:.0f}%) — watch for support or further breakdown.")

    # Relative strength
    rs = latest.get("relative_strength")
    if pd.notna(rs):
        if rs > 0.03:
            lines.append(f"The asset is <b>outperforming SPY</b> by {rs:.1%} over 20 days — showing relative leadership.")
        elif rs < -0.03:
            lines.append(f"The asset is <b>underperforming SPY</b> by {abs(rs):.1%} over 20 days — lagging the market.")

    if not lines:
        return "Signal data is limited — try a ticker with more history for a fuller picture."

    return " ".join(lines)
