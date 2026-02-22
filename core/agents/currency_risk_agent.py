from __future__ import annotations
from typing import Any, Dict, Optional, List
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class CurrencyRiskAgent(AgentBase):
    """
    Agent responsible for assessing Currency Risk (FX).
    Evaluates exposure to foreign exchange rate fluctuations.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, fx_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess currency risk.

        Args:
            fx_data: Dictionary containing:
                - reporting_currency (str)
                - foreign_revenue_pct (float)
                - foreign_debt_pct (float)
                - major_exposures (List[str], e.g. ["EUR", "JPY"])
                - hedging_ratio (float, optional)

        Returns:
            Dict containing currency risk assessment.
        """
        logger.info("Starting Currency Risk Assessment...")

        foreign_rev = fx_data.get("foreign_revenue_pct", 0.0)
        foreign_debt = fx_data.get("foreign_debt_pct", 0.0)
        hedging = fx_data.get("hedging_ratio", 0.0)

        # Unhedged Exposure
        net_exposure = (foreign_rev + foreign_debt) * (1 - hedging)

        risk_score = 0
        if net_exposure > 0.50:
            risk_score = 80
        elif net_exposure > 0.25:
            risk_score = 50
        elif net_exposure > 0.10:
            risk_score = 30
        else:
            risk_score = 10

        result = {
            "currency_risk_score": float(risk_score),
            "net_exposure_pct": float(net_exposure),
            "risk_level": "High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low"
        }

        logger.info(f"Currency Risk Assessment Complete: {result}")
        return result
