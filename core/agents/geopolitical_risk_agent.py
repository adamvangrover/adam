# core/agents/geopolitical_risk_agent.py

from __future__ import annotations
from typing import Any, Dict, Optional, List
import logging
from core.agents.agent_base import AgentBase
from core.utils.data_utils import send_message

logger = logging.getLogger(__name__)

class GeopoliticalRiskAgent(AgentBase):
    """
    Agent responsible for assessing Geopolitical Risk.
    Evaluates political stability, trade relations, and conflict risks.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.data_sources = config.get('data_sources', {})

    async def execute(self, regions: List[str]) -> Dict[str, Any]:
        """
        Assess geopolitical risks for a list of regions.

        Args:
            regions: List of country codes or region names (e.g., ["US", "CN", "EU"]).

        Returns:
             Dict with risk assessment per region and global score.
        """
        logger.info(f"Assessing geopolitical risks for: {regions}")

        risk_assessments = {}
        global_score = 0

        # Handle single string input if passed accidentally
        if isinstance(regions, str):
            regions = [regions]

        for region in regions:
            score = self.calculate_political_risk_index(region)
            key_risks = self.identify_key_risks(region)
            risk_assessments[region] = {
                "risk_index": score,
                "key_risks": key_risks,
                "level": "High" if score > 70 else "Medium" if score > 40 else "Low"
            }
            global_score = max(global_score, score) # Max risk drives global concern

        result = {
            'global_risk_index': global_score,
            'regional_assessments': risk_assessments
        }

        # Send risk assessments to message queue (legacy support)
        try:
            message = {'agent': 'geopolitical_risk_agent', 'risk_assessments': result}
            send_message(message)
        except Exception as e:
            logger.warning(f"Failed to send legacy message: {e}")

        return result

    def calculate_political_risk_index(self, region: str) -> int:
        """
        Calculates a political risk index (0-100).
        """
        # Mock logic - in real world would fetch from API
        risk_map = {
            "US": 30,
            "EU": 40,
            "CN": 60,
            "RU": 90,
            "EM": 70, # Emerging Markets
            "Middle East": 80
        }
        return risk_map.get(region, 50)

    def identify_key_risks(self, region: str) -> List[str]:
        """
        Identifies key risks for a region.
        """
        risks = []
        if region in ["US", "CN"]:
            risks.append("Trade Tensions")
        if region in ["RU", "Middle East"]:
            risks.append("Conflict")
        if region == "EU":
            risks.append("Regulatory Fragmentation")

        if not risks:
            risks.append("General Uncertainty")

        return risks

    def assess_geopolitical_risks(self) -> Dict[str, Any]:
        """
        Legacy method for assessing geopolitical risks.
        Wrapper for backward compatibility with older tests/workflows.
        """
        logger.info("Assessing geopolitical risks (Legacy Mode)...")

        # Default/Global assessment logic
        default_region = "Global"
        score = self.calculate_political_risk_index(default_region)
        key_risks = self.identify_key_risks(default_region)

        risk_assessments = {
            'political_risk_index': score,
            'key_risks': key_risks,
            # Add other keys if expected by legacy consumers
        }

        try:
            message = {'agent': 'geopolitical_risk_agent', 'risk_assessments': risk_assessments}
            send_message(message)
        except Exception as e:
            logger.warning(f"Failed to send legacy message: {e}")

        return risk_assessments
