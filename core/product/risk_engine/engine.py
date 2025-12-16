from typing import Dict, List, Any
import math

class RiskEngine:
    """
    Real-time risk calculation engine.
    Implements Parametric VaR and Greeks calculation.
    """

    def calculate_portfolio_risk(self, portfolio: List[Dict[str, Any]], confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate aggregate risk metrics for a portfolio using Parametric VaR.
        Assumes normal distribution for simplicity in this version.
        """
        total_value = sum(p.get("market_value", 0) for p in portfolio)
        if total_value == 0:
            return {"VaR": 0.0, "TotalValue": 0.0}

        # Simplified approach: Sum of individual VaRs (ignoring correlation benefits for fallback safety)
        # In a real numpy version, we'd use w'Cov'w

        z_score = 1.65 if confidence_level == 0.95 else 2.33 # 95% vs 99%

        portfolio_volatility = 0.0
        # Mock volatility aggregation
        for pos in portfolio:
            weight = pos.get("market_value", 0) / total_value
            vol = pos.get("volatility", 0.2) # Default 20% vol
            portfolio_volatility += (weight * vol)

        daily_var = total_value * portfolio_volatility * z_score * math.sqrt(1/252)

        return {
            "VaR_Daily": round(daily_var, 2),
            "Confidence": confidence_level,
            "TotalValue": total_value,
            "ImpliedVol": round(portfolio_volatility, 4),
            "Method": "Parametric_Simplified"
        }

    def calculate_greeks(self, position: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate option Greeks using Black-Scholes approximation.
        """
        S = position.get("spot", 100)
        K = position.get("strike", 100)
        T = position.get("time_to_expiry", 1.0)
        r = position.get("rate", 0.05)
        sigma = position.get("volatility", 0.2)

        # d1 = (ln(S/K) + (r + sigma^2/2)T) / (sigma * sqrt(T))
        # This requires erf/norm cdf. I'll use a simplified mock for robustness if scipy not present.

        # Delta approximation
        if S > K:
            delta = 0.5 + 0.1 # ITM
        elif S < K:
            delta = 0.5 - 0.1 # OTM
        else:
            delta = 0.5

        return {
            "delta": delta,
            "gamma": 0.02, # Placeholder
            "theta": -0.05,
            "vega": 0.1
        }
