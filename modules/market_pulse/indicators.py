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
