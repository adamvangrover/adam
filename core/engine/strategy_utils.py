import logging
from typing import Any, Dict, Literal

logger = logging.getLogger(__name__)

def determine_ma_posture(financials: Dict[str, Any], market_data: Dict[str, Any]) -> Literal["Buyer", "Seller", "Neutral"]:
    """
    Determines if the entity is likely an Acquirer or a Target.
    """
    cash = financials.get("cash_equivalents", 0)
    market_cap = market_data.get("market_cap", 1000)
    debt = financials.get("total_debt", 0)

    cash_to_mcap = cash / market_cap if market_cap else 0
    leverage = debt / financials.get("ebitda", 1) if financials.get("ebitda") else 0

    if cash_to_mcap > 0.15 and leverage < 2.0:
        return "Buyer" # Cash rich, low debt
    elif market_cap < 5000 and leverage < 3.0: # Small cap, reasonable debt
        return "Target"
    else:
        return "Neutral"

def synthesize_verdict(
    valuation: Dict[str, Any],
    credit_rating: str,
    risk_score: float, # 0-10, 10 is high risk
    ma_posture: str
) -> Dict[str, Any]:
    """
    Synthesizes all analyses into a final conviction level and recommendation.
    """
    score = 5 # Neutral start

    # 1. Valuation Impact
    if valuation.get("verdict") == "Undervalued":
        score += 2
    elif valuation.get("verdict") == "Overvalued":
        score -= 2

    # 2. Credit Impact
    if credit_rating == "Pass":
        score += 1
    elif credit_rating in ["Substandard", "Doubtful"]:
        score -= 3

    # 3. Risk Impact (Lower risk is better for Long)
    if risk_score > 7.0:
        score -= 2
    elif risk_score < 3.0:
        score += 1

    # 4. M&A Catalyst
    if ma_posture == "Target":
        score += 1 # Speculative premium

    # Normalize 1-10
    score = max(1, min(10, score))

    recommendation = "Hold"
    if score >= 8:
        recommendation = "Strong Buy"
    elif score >= 6:
        recommendation = "Buy"
    elif score <= 3:
        recommendation = "Sell"

    return {
        "conviction_level": int(score),
        "recommendation": recommendation,
        "time_horizon": "12-18 Months",
        "rationale_summary": f"Rated {recommendation} with Conviction {score}/10. Driven by {valuation.get('verdict')} valuation and {credit_rating} credit profile."
    }
