from __future__ import annotations
from typing import Any, Dict, Optional, List
import logging
import numpy as np
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class CurrencyRiskAgent(AgentBase):
    """
    Agent responsible for assessing Currency Risk (FX).
    Evaluates Portfolio VaR, Interest Rate Parity deviations, and Unhedged Exposure.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, fx_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess currency risk.

        Args:
            fx_data: Dictionary containing:
                - reporting_currency (str)
                - exposures: Dict[str, float] (e.g. {"EUR": 100000, "JPY": 500000}) - Net Amounts
                - portfolio_value: float (Total value in reporting currency)
                - hedging_ratio (float, optional)
                - correlations: Dict (optional) or assume default
                - market_rates: Dict (optional) containing spot, forward, interest rates for IRP check.
                  e.g. {"EUR": {"spot": 1.1, "forward_1y": 1.12, "rate_domestic": 0.05, "rate_foreign": 0.03}}

        Returns:
            Dict containing currency risk assessment.
        """
        logger.info("Starting Currency Risk Assessment...")

        exposures = fx_data.get("exposures", {})
        portfolio_val = fx_data.get("portfolio_value", 1.0)
        hedging = fx_data.get("hedging_ratio", 0.0)

        # 1. Unhedged Exposure Calculation
        total_gross_exposure = sum(abs(v) for v in exposures.values())
        unhedged_gross = total_gross_exposure * (1 - hedging)
        exposure_pct = unhedged_gross / portfolio_val if portfolio_val > 0 else 0

        risk_score = 0
        if exposure_pct > 0.50: risk_score += 50
        elif exposure_pct > 0.25: risk_score += 30
        elif exposure_pct > 0.10: risk_score += 10

        # 2. Portfolio VaR (Parametric)
        # Assume 10% annual volatility for all pairs if not provided
        # Assume 0.3 correlation between pairs
        vol_assumption = 0.10
        corr_assumption = 0.30

        # Construct weight vector
        currencies = list(exposures.keys())
        weights = np.array([exposures[c] * (1-hedging) for c in currencies])

        if len(weights) > 0:
            # Covariance Matrix construction
            n = len(weights)
            cov_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        cov_matrix[i, j] = vol_assumption**2
                    else:
                        cov_matrix[i, j] = vol_assumption * vol_assumption * corr_assumption

            # Portfolio Variance = w.T * Cov * w
            port_var_val = weights.T @ cov_matrix @ weights
            port_std_val = np.sqrt(port_var_val)

            # 95% VaR (1.645 sigma)
            fx_var_95 = 1.645 * port_std_val
            fx_var_pct = fx_var_95 / portfolio_val if portfolio_val > 0 else 0
        else:
            fx_var_95 = 0.0
            fx_var_pct = 0.0

        if fx_var_pct > 0.05: risk_score += 30
        elif fx_var_pct > 0.02: risk_score += 15

        # 3. Interest Rate Parity (IRP) Check
        market_rates = fx_data.get("market_rates", {})
        irp_deviations = {}

        for ccy, rates in market_rates.items():
            spot = rates.get("spot")
            fwd_mkt = rates.get("forward_1y")
            rd = rates.get("rate_domestic")
            rf = rates.get("rate_foreign")

            if all(v is not None for v in [spot, fwd_mkt, rd, rf]):
                # Theoretical Forward F = S * (1+rd)/(1+rf)
                fwd_theo = spot * (1 + rd) / (1 + rf)

                diff_pct = (fwd_mkt - fwd_theo) / fwd_theo
                irp_deviations[ccy] = {
                    "theoretical_fwd": float(fwd_theo),
                    "market_fwd": float(fwd_mkt),
                    "deviation_pct": float(diff_pct),
                    "signal": "Arbitrage Opportunity" if abs(diff_pct) > 0.01 else "Parity Holds"
                }

                if abs(diff_pct) > 0.02: # Significant deviation
                    risk_score += 10 # Market dislocation risk

        risk_score = min(100.0, risk_score)
        level = "High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low"

        result = {
            "currency_risk_score": float(risk_score),
            "risk_level": level,
            "exposure_metrics": {
                "total_unhedged_exposure": float(unhedged_gross),
                "exposure_pct_of_portfolio": float(exposure_pct),
                "portfolio_fx_var_95": float(fx_var_95),
                "portfolio_fx_var_pct": float(fx_var_pct)
            },
            "irp_analysis": irp_deviations
        }

        logger.info(f"Currency Risk Assessment Complete: {result}")
        return result
