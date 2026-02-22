from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class EconomicRiskAgent(AgentBase):
    """
    Agent responsible for assessing Economic Risk (Macro).
    Evaluates recession risk, stagflation, and interest rate sensitivity.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess economic risk.

        Args:
            macro_data: Dictionary containing:
                - gdp_growth (float, %)
                - inflation_rate (float, %)
                - unemployment_rate (float, %)
                - yield_curve_inversion (bool)

        Returns:
            Dict containing economic risk assessment.
        """
        logger.info("Starting Economic Risk Assessment...")

        gdp = macro_data.get("gdp_growth", 0.02)
        inflation = macro_data.get("inflation_rate", 0.02)
        unemp = macro_data.get("unemployment_rate", 0.04)
        inverted_yield = macro_data.get("yield_curve_inversion", False)

        risk_score = 0

        # Recession Risk
        if gdp < 0: risk_score += 40 # Recession
        elif gdp < 0.01: risk_score += 20 # Slow growth

        # Stagflation Risk (Low Growth + High Inflation)
        if gdp < 0.01 and inflation > 0.05:
            risk_score += 30
            scenario = "Stagflation"
        elif gdp > 0.03 and inflation > 0.05:
            risk_score += 10
            scenario = "Overheating"
        else:
            scenario = "Normal"

        # Yield Curve signal
        if inverted_yield:
            risk_score += 20

        risk_score = min(100.0, risk_score)
        level = "High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low"

        result = {
            "economic_risk_score": float(risk_score),
            "risk_level": level,
            "scenario": scenario,
            "factors": {
                "recession_risk": "High" if gdp < 0 else "Low",
                "inflation_risk": "High" if inflation > 0.05 else "Low",
                "yield_curve_signal": "Warning" if inverted_yield else "Normal"
            }
        }

        logger.info(f"Economic Risk Assessment Complete: {result}")
        return result
