import pandas as pd
from datetime import datetime, timedelta
from modules.data.fetcher import fetch_prices, fetch_fred

_THREE_YEAR_DAYS = 1095  # ~3 calendar years


def _start() -> str:
    return (datetime.today() - timedelta(days=_THREE_YEAR_DAYS)).strftime("%Y-%m-%d")


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def get_yield_curve() -> dict:
    """
    Fetch 2Y and 10Y treasury yields (FRED: DGS2, DGS10), return 10Y–2Y spread.
    Positive = normal/calm, negative = inverted (recession warning).
    Score with invert=True (high spread = calm).
    """
    start, end = _start(), _today()
    dgs2  = fetch_fred("DGS2", start, end)
    dgs10 = fetch_fred("DGS10", start, end)
    spread = (dgs10 - dgs2).dropna()
    return {
        "series": spread,
        "current": float(spread.iloc[-1]),
        "label": "Yield Curve (10Y–2Y)",
        "unit": "%",
        "invert": True,
    }


def get_fed_funds() -> dict:
    """
    Fetch Fed Funds Rate (FRED: FEDFUNDS).
    High rate = tighter financial conditions = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    series = fetch_fred("FEDFUNDS", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Fed Funds Rate",
        "unit": "%",
        "invert": False,
    }


def get_vix() -> dict:
    """
    Fetch VIX equity fear index (yfinance: ^VIX).
    High = stressed. Score with invert=False.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["^VIX"], start, end)
    series = prices["^VIX"].dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "VIX",
        "unit": "pts",
        "invert": False,
    }


def get_aaii_sentiment() -> dict:
    """
    Fetch AAII bull/bear survey (FRED: AAIIBULL, AAIIBEAR), return bull-bear spread.
    Low spread = fearful retail = stressed. High spread = bullish = calm.
    Score with invert=True. Note: weekly series.
    """
    start, end = _start(), _today()
    bull = fetch_fred("AAIIBULL", start, end)
    bear = fetch_fred("AAIIBEAR", start, end)
    combined = pd.concat([bull.rename("bull"), bear.rename("bear")], axis=1).dropna()
    spread = (combined["bull"] - combined["bear"]).rename("spread")
    return {
        "series": spread,
        "current": float(spread.iloc[-1]),
        "label": "AAII Bull–Bear Spread",
        "unit": "%",
        "invert": True,
    }


def get_consumer_confidence() -> dict:
    """
    Fetch U of Michigan Consumer Sentiment (FRED: UMCSENT).
    High = confident = calm. Score with invert=True. Note: monthly series.
    """
    start, end = _start(), _today()
    series = fetch_fred("UMCSENT", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Consumer Confidence (UMich)",
        "unit": "index",
        "invert": True,
    }


def get_hy_credit_spread() -> dict:
    """
    Fetch US HY option-adjusted spread (FRED: BAMLH0A0HYM2).
    High spread = risky borrowing is expensive = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    series = fetch_fred("BAMLH0A0HYM2", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "HY Credit Spread",
        "unit": "%",
        "invert": False,
    }


def get_gold_spy_ratio() -> dict:
    """
    Compute GLD / SPY price ratio (yfinance: GLD, SPY).
    High ratio = flight to gold = risk-off = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["GLD", "SPY"], start, end)
    ratio = (prices["GLD"] / prices["SPY"]).dropna()
    return {
        "series": ratio,
        "current": float(ratio.iloc[-1]),
        "label": "Gold / SPY Ratio",
        "unit": "ratio",
        "invert": False,
    }


def get_dxy() -> dict:
    """
    Fetch Dollar Index (yfinance: DX-Y.NYB).
    Strong dollar = global risk-off = stressed.
    Score with invert=False.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["DX-Y.NYB"], start, end)
    series = prices["DX-Y.NYB"].dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Dollar Index (DXY)",
        "unit": "index",
        "invert": False,
    }


HEATMAP_TICKERS = ["SPY", "^FTSE", "^GDAXI", "^N225", "^HSI", "EEM"]
HEATMAP_LABELS = {
    "SPY":    "S&P 500 (US)",
    "^FTSE":  "FTSE 100 (UK)",
    "^GDAXI": "DAX (Germany)",
    "^N225":  "Nikkei 225 (Japan)",
    "^HSI":   "Hang Seng (HK)",
    "EEM":    "Emerging Markets",
}


def get_global_heatmap() -> pd.DataFrame:
    """
    Fetch 40-day price history for global indices (yfinance).
    Returns DataFrame: ticker, label, 1d return, 1w return, 1m return.
    Uses a 40-day window — enough for 1-month + buffer on short weeks/holidays.
    """
    end = _today()
    start = (datetime.today() - timedelta(days=40)).strftime("%Y-%m-%d")
    prices = fetch_prices(HEATMAP_TICKERS, start, end)

    rows = []
    for ticker in HEATMAP_TICKERS:
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        r1d = float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) >= 2 else None
        r1w = float(s.iloc[-1] / s.iloc[-6] - 1) if len(s) >= 6 else None
        r1m = float(s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 20 else None
        rows.append({
            "ticker": ticker,
            "label": HEATMAP_LABELS.get(ticker, ticker),
            "1d": r1d,
            "1w": r1w,
            "1m": r1m,
        })
    return pd.DataFrame(rows)


def get_vix_term_structure() -> dict:
    """
    Fetch VIX9D (^VIX9D), VIX (^VIX), VIX3M (^VIX3M) from yfinance.

    Contango:      VIX9D < VIX < VIX3M  — normal, market calm
    Backwardation: VIX9D > VIX > VIX3M  — panic, near-term fear elevated

    Returns current values for all three + structure label.
    Note: VIX3M was formerly ^VXV; yfinance may serve either ticker.
    """
    start, end = _start(), _today()
    prices = fetch_prices(["^VIX9D", "^VIX", "^VIX3M"], start, end)

    def _last(col):
        if col not in prices.columns:
            return None
        s = prices[col].dropna()
        return float(s.iloc[-1]) if len(s) > 0 else None

    vix9d = _last("^VIX9D")
    vix   = _last("^VIX")
    vix3m = _last("^VIX3M")

    if vix9d is not None and vix3m is not None:
        structure = "Backwardation (Panic)" if vix9d > vix3m else "Contango (Calm)"
    else:
        structure = "Unknown"

    return {
        "vix9d": vix9d,
        "vix": vix,
        "vix3m": vix3m,
        "structure": structure,
        "label": "VIX Term Structure",
    }


def get_m2() -> dict:
    """
    Fetch M2 money supply (FRED: M2SL). Monthly series, units in billions USD.
    High/rising M2 = more liquidity = calm market conditions.
    Score with invert=True.
    """
    start, end = _start(), _today()
    series = fetch_fred("M2SL", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "M2 Money Supply",
        "unit": "$B",
        "invert": True,
    }


def get_fed_balance_sheet() -> dict:
    """
    Fetch Fed balance sheet total assets (FRED: WALCL). Weekly series, units in millions USD.
    Shrinking balance sheet = tighter liquidity = stressed.
    Score with invert=True (large balance sheet = more liquidity = calm).
    """
    start, end = _start(), _today()
    series = fetch_fred("WALCL", start, end).dropna()
    return {
        "series": series,
        "current": float(series.iloc[-1]),
        "label": "Fed Balance Sheet",
        "unit": "$M",
        "invert": True,
    }
