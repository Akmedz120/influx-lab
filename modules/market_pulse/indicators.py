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
