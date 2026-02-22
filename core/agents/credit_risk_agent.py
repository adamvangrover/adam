from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class CreditRiskAgent(AgentBase):
    """
    Agent responsible for assessing Credit Risk (Default Risk).
    Calculates Altman Z-Score and estimates Default Probability.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the credit risk assessment.

        Args:
            financial_data: Dictionary containing financial metrics:
                - working_capital
                - retained_earnings
                - ebit
                - total_assets
                - market_value_equity
                - total_liabilities
                - sales

        Returns:
            Dict containing credit risk metrics.
        """
        logger.info("Starting Credit Risk Assessment...")

        try:
            # Extract inputs with defaults to 0 to prevent crashes, but validation is better
            wc = financial_data.get("working_capital", 0)
            re = financial_data.get("retained_earnings", 0)
            ebit = financial_data.get("ebit", 0)
            ta = financial_data.get("total_assets", 1) # Avoid div by zero
            mve = financial_data.get("market_value_equity", 0)
            tl = financial_data.get("total_liabilities", 1) # Avoid div by zero
            sales = financial_data.get("sales", 0)

            if ta == 0: ta = 1
            if tl == 0: tl = 1

            # Altman Z-Score Calculation (for public manufacturing, can be adapted)
            # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
            # A = Working Capital / Total Assets
            # B = Retained Earnings / Total Assets
            # C = EBIT / Total Assets
            # D = Market Value of Equity / Total Liabilities
            # E = Sales / Total Assets

            A = wc / ta
            B = re / ta
            C = ebit / ta
            D = mve / tl
            E = sales / ta

            z_score = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)

            # Interpret Z-Score
            if z_score > 2.99:
                zone = "Safe"
                default_prob = "Low (< 5%)"
            elif z_score > 1.81:
                zone = "Grey"
                default_prob = "Moderate (5-20%)"
            else:
                zone = "Distress"
                default_prob = "High (> 20%)"

            result = {
                "z_score": float(z_score),
                "zone": zone,
                "default_probability_estimate": default_prob,
                "components": {
                    "A": float(A), "B": float(B), "C": float(C), "D": float(D), "E": float(E)
                }
            }

            logger.info(f"Credit Risk Assessment Complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error calculating credit risk: {e}")
            return {"error": str(e)}
