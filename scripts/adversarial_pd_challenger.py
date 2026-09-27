import math
from scipy.stats import norm
import json
import argparse

def calculate_forward_looking_pd(
    name: str = "Unknown",
    ticker: str = "UNKNOWN",
    current_debt: float = 0.0,
    short_term_debt: float = 0.0,
    projected_fcf: float = 0.0,
    implied_volatility: float = 0.0,
    forward_interest_rate: float = 0.0,
    equity_value: float = 0.0,
    macro_stress_factor: float = 1.0
) -> dict:
    current_debt = float(current_debt)
    short_term_debt = float(short_term_debt)
    projected_fcf = float(projected_fcf)
    implied_volatility = float(implied_volatility)
    forward_interest_rate = float(forward_interest_rate)
    equity_value = float(equity_value)
    macro_stress_factor = float(macro_stress_factor)

    total_liability = current_debt
    v_firm = equity_value + total_liability
    v_firm_stressed = v_firm * macro_stress_factor
    horizon = 1.0

    if v_firm_stressed > 0 and implied_volatility > 0:
        long_term_debt = total_liability - short_term_debt
        default_point = short_term_debt + 0.5 * long_term_debt
        if default_point > 0:
            numerator = math.log(v_firm_stressed / default_point) + (forward_interest_rate - 0.5 * implied_volatility**2) * horizon
            denominator = implied_volatility * math.sqrt(horizon)
            dd_forward = numerator / denominator
            pd_structural = norm.cdf(-dd_forward)
        else:
            dd_forward = 0.0
            pd_structural = 0.0
    else:
        dd_forward = 0.0
        pd_structural = 1.0

    refi_need = short_term_debt - projected_fcf
    if refi_need > 0 and v_firm_stressed > 0:
        liquidity_stress = (refi_need / v_firm_stressed) * (1 + forward_interest_rate)
    elif v_firm_stressed <= 0:
        liquidity_stress = float('inf')
    else:
        liquidity_stress = 0.0

    if liquidity_stress == float('inf'):
        pd_liquidity = 1.0
    else:
        pd_liquidity = 1.0 - math.exp(-liquidity_stress)

    pd_challenger = 0.7 * pd_structural + 0.3 * pd_liquidity
    pd_challenger = max(0.0001, min(0.9999, pd_challenger))

    return {
        "entity": {
            "name": name,
            "ticker": ticker
        },
        "inputs": {
            "v_firm_stressed": v_firm_stressed,
            "implied_volatility": implied_volatility,
            "forward_interest_rate": forward_interest_rate,
            "projected_fcf": projected_fcf,
            "macro_stress_factor": macro_stress_factor
        },
        "components": {
            "structural_pd": pd_structural,
            "liquidity_pd": pd_liquidity,
            "forward_distance_to_default": dd_forward
        },
        "adversarial_pd_challenger": pd_challenger
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Looking PD Adversarial Challenger")
    parser.add_argument("--name", type=str, default="Unknown")
    parser.add_argument("--ticker", type=str, default="UNKNOWN")
    parser.add_argument("--current_debt", type=float, default=5000.0)
    parser.add_argument("--short_term_debt", type=float, default=1000.0)
    parser.add_argument("--projected_fcf", type=float, default=200.0)
    parser.add_argument("--implied_volatility", type=float, default=0.40)
    parser.add_argument("--forward_interest_rate", type=float, default=0.05)
    parser.add_argument("--equity_value", type=float, default=3000.0)
    parser.add_argument("--macro_stress_factor", type=float, default=0.85)

    args = parser.parse_args()
    res = calculate_forward_looking_pd(
        name=args.name,
        ticker=args.ticker,
        current_debt=args.current_debt,
        short_term_debt=args.short_term_debt,
        projected_fcf=args.projected_fcf,
        implied_volatility=args.implied_volatility,
        forward_interest_rate=args.forward_interest_rate,
        equity_value=args.equity_value,
        macro_stress_factor=args.macro_stress_factor
    )
    print(json.dumps(res, indent=2))
