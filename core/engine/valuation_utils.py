from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def calculate_dcf(financials: Dict[str, Any], risk_free_rate: float = 0.04, scenario: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Performs a simplified Discounted Cash Flow (DCF) analysis.

    Args:
        financials: Dict containing 'fcf' (Free Cash Flow), 'growth_rate', 'beta'.
        risk_free_rate: The current risk-free rate (e.g., 10y Treasury).
        scenario: Optional dict for Scenario Injection (e.g., {'growth_rate': 0.02, 'market_risk_premium': 0.08})

    Returns:
        Dict with 'wacc', 'terminal_growth', 'intrinsic_value', 'share_price'.
    """
    logger.info("Running DCF Analysis...")

    if scenario:
        logger.info(f"Injecting Scenario: {scenario}")

    # Extract or Default (Scenario overrides financials)
    scenario = scenario or {}

    fcf = scenario.get("fcf", financials.get("fcf", 1000))
    growth_rate = scenario.get("growth_rate", financials.get("growth_rate", 0.05))
    beta = scenario.get("beta", financials.get("beta", 1.2))
    shares_outstanding = financials.get("shares_outstanding", 500)

    # Market Assumptions (Scenario overrides defaults)
    market_risk_premium = scenario.get("market_risk_premium", 0.05)
    cost_of_debt = scenario.get("cost_of_debt", 0.06)
    tax_rate = scenario.get("tax_rate", 0.21)

    # Capital Structure
    debt_equity_ratio = financials.get("debt_equity_ratio", 0.5)

    # 1. Calculate WACC
    # Scenario injection for risk_free_rate is handled by the argument, but check scenario dict too
    rfr = scenario.get("risk_free_rate", risk_free_rate)

    cost_of_equity = rfr + beta * market_risk_premium
    wacc = (cost_of_equity * (1 / (1 + debt_equity_ratio))) + \
           (cost_of_debt * (1 - tax_rate) * (debt_equity_ratio / (1 + debt_equity_ratio)))

    # 2. Project FCF (5 years)
    projected_fcf = []
    current_fcf = fcf
    for _ in range(5):
        current_fcf *= (1 + growth_rate)
        projected_fcf.append(current_fcf)

    # 3. Terminal Value
    terminal_growth = 0.025
    terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)

    # 4. Discount to Present
    enterprise_value = 0
    for t, cash_flow in enumerate(projected_fcf, 1):
        enterprise_value += cash_flow / ((1 + wacc) ** t)

    enterprise_value += terminal_value / ((1 + wacc) ** 5)

    # 5. Equity Value
    net_debt = financials.get("net_debt", 2000)
    equity_value = enterprise_value - net_debt
    intrinsic_share_price = equity_value / shares_outstanding

    return {
        "wacc": round(wacc, 4),
        "terminal_growth": terminal_growth,
        "intrinsic_value": round(equity_value, 2),
        "intrinsic_share_price": round(intrinsic_share_price, 2),
        "method": "5-Year DCF / Gordon Growth"
    }

def calculate_multiples(financials: Dict[str, Any], peer_group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs Relative Valuation using trading multiples.
    """
    logger.info("Running Multiples Analysis...")

    ev = financials.get("enterprise_value", 10000)
    ebitda = financials.get("ebitda", 2000)

    current_ev_ebitda = ev / ebitda if ebitda else 0

    # Peer Analysis
    peer_multiples = [p.get("ev_ebitda", 10.0) for p in peer_group]
    peer_median = sum(peer_multiples) / len(peer_multiples) if peer_multiples else 10.0

    premium_discount = (current_ev_ebitda - peer_median) / peer_median

    return {
        "current_ev_ebitda": round(current_ev_ebitda, 2),
        "peer_median_ev_ebitda": round(peer_median, 2),
        "premium_discount_pct": round(premium_discount * 100, 2),
        "verdict": "Overvalued" if premium_discount > 0.1 else "Undervalued" if premium_discount < -0.1 else "Fairly Valued"
    }

def get_price_targets(intrinsic_price: float, volatility: float) -> Dict[str, float]:
    """
    Generates Bull, Base, and Bear case price targets.
    """
    return {
        "bear_case": round(intrinsic_price * (1 - volatility * 2), 2),
        "base_case": round(intrinsic_price, 2),
        "bull_case": round(intrinsic_price * (1 + volatility * 2), 2)
    }
