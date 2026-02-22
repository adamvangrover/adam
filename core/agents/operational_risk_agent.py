from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class OperationalRiskAgent(AgentBase):
    """
    Agent responsible for assessing Operational Risk.
    Evaluates risks related to internal processes, people, and systems.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

    async def execute(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the operational risk assessment.

        Args:
            company_profile: Dictionary containing:
                - years_in_business
                - employee_turnover_rate
                - compliance_incidents (count)
                - management_tenure (avg years)
                - it_downtime_hours (annual)

        Returns:
            Dict containing operational risk assessment.
        """
        logger.info("Starting Operational Risk Assessment...")

        # Default values - reasonable assumptions if data missing
        years = company_profile.get("years_in_business", 5)
        turnover = company_profile.get("employee_turnover_rate", 0.10)
        incidents = company_profile.get("compliance_incidents", 0)
        mgmt_tenure = company_profile.get("management_tenure", 5)

        risk_score = 0.0

        # Heuristic scoring (0 = Low Risk, 100 = High Risk)

        # 1. Stability (Years)
        if years < 2: risk_score += 20
        elif years < 5: risk_score += 10

        # 2. People (Turnover)
        if turnover > 0.20: risk_score += 20
        elif turnover > 0.15: risk_score += 10

        # 3. Compliance
        risk_score += (incidents * 10) # 10 points per incident

        # 4. Leadership
        if mgmt_tenure < 2: risk_score += 15

        risk_score = min(100.0, risk_score)

        level = "Low"
        if risk_score > 60: level = "High"
        elif risk_score > 30: level = "Medium"

        result = {
            "operational_risk_score": float(risk_score),
            "risk_level": level,
            "factors": {
                "stability": "Low" if years < 5 else "High",
                "compliance_history": "Clean" if incidents == 0 else "Issues",
                "workforce_stability": "Low" if turnover > 0.20 else "High"
            }
        }

        logger.info(f"Operational Risk Assessment Complete: {result}")
        return result
