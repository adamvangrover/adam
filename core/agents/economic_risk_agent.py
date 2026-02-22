from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class EconomicRiskAgent(AgentBase):
    """
    Agent responsible for assessing Economic Risk (Macro).
    Evaluates recession risk, stagflation, Phillips Curve deviations, and Misery Index.
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

        risk_score, scenario, misery_index = self._analyze_macro_scenario(gdp, inflation, unemp, inverted_yield)

        level = "High" if risk_score > 60 else "Medium" if risk_score > 30 else "Low"

        result = {
            "economic_risk_score": float(risk_score),
            "risk_level": level,
            "macro_scenario": scenario,
            "metrics": {
                "misery_index": float(misery_index),
                "misery_classification": self._classify_misery(misery_index),
                "yield_curve_signal": "Warning" if inverted_yield else "Normal"
            },
            "input_data": {
                "gdp_growth": gdp,
                "inflation_rate": inflation,
                "unemployment_rate": unemp
            }
        }

        logger.info(f"Economic Risk Assessment Complete: {result}")
        return result

    def _analyze_macro_scenario(self, gdp, inflation, unemployment, yield_inverted):
        """Analyzes the macro scenario and calculates a risk score."""
        misery_index = inflation + unemployment

        score = 0
        scenario = "Normal Expansion"

        # 1. Misery Index Component (Inflation + Unemployment)
        # Range logic:
        # < 6%: Great (0)
        # 6-9%: Normal (10)
        # 9-12%: Concerning (30)
        # 12-16%: High (50)
        # > 16%: Crisis (70)

        if misery_index > 0.16: score += 70
        elif misery_index > 0.12: score += 50
        elif misery_index > 0.09: score += 30
        elif misery_index > 0.06: score += 10

        # 2. GDP Growth Component & Scenario Labeling
        if gdp < -0.02:
            score += 30
            scenario = "Deep Recession"
        elif gdp < 0.00:
            score += 20
            scenario = "Recession"
        elif gdp < 0.015:
            # Low Growth
            if inflation > 0.04:
                scenario = "Stagflation"
                score += 20 # Additional penalty for stagflation
            else:
                scenario = "Stagnation"
                score += 10
        elif gdp > 0.04:
            # High Growth
            if inflation > 0.04:
                scenario = "Overheating"
                score += 10
            else:
                scenario = "Boom"

        # 3. Yield Curve Component
        if yield_inverted:
            score += 20
            if scenario == "Normal Expansion":
                scenario = "Late Cycle (Yield Inversion)"

        return min(100, score), scenario, misery_index

    def _classify_misery(self, misery_index):
        if misery_index < 0.06: return "Low"
        if misery_index < 0.10: return "Moderate"
        if misery_index < 0.15: return "High"
        return "Severe"
