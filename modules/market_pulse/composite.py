import sys as _sys
from modules.market_pulse.indicators import (
    get_vix, get_hy_credit_spread, get_dxy,
    get_yield_curve, get_gold_spy_ratio, get_aaii_sentiment,
)
from modules.market_pulse.scoring import percentile_score

# Names looked up at call time so unittest.mock.patch works correctly.
_COMPOSITE_FN_NAMES = [
    "get_vix", "get_hy_credit_spread", "get_dxy",
    "get_yield_curve", "get_gold_spy_ratio", "get_aaii_sentiment",
]


def compute_fear_greed_index() -> dict:
    """
    Custom Fear & Greed Index built from 6 Phase 2a indicators.

    Each indicator is scored 0–100 via 3-year percentile rank.
    Equal-weighted average gives the composite score.

    Score interpretation:
        0–20   : Extreme Greed (market very calm)
        20–40  : Greed
        40–60  : Neutral
        60–80  : Fear
        80–100 : Extreme Fear (market very stressed)

    Failed/unavailable indicators are skipped; remaining are averaged.
    Returns {"score": float, "label": str, "details": list[dict]}
    """
    _mod = _sys.modules[__name__]
    scores = []
    details = []
    for name in _COMPOSITE_FN_NAMES:
        try:
            fn = getattr(_mod, name)
            ind = fn()
            sc = percentile_score(ind["series"], ind["current"], invert=ind.get("invert", False))
            scores.append(sc["score"])
            details.append({
                "label": ind["label"],
                "score": sc["score"],
                "color": sc["color"],
            })
        except Exception:
            pass

    if not scores:
        return {"score": 50.0, "label": "Neutral", "details": []}

    avg = round(sum(scores) / len(scores), 1)

    if avg >= 80:
        label = "Extreme Fear"
    elif avg >= 60:
        label = "Fear"
    elif avg >= 40:
        label = "Neutral"
    elif avg >= 20:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {"score": avg, "label": label, "details": details}
