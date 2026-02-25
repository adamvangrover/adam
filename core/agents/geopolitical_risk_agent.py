# core/agents/geopolitical_risk_agent.py

from __future__ import annotations
from typing import Any, Dict, Optional, List, Union
import logging
from core.agents.agent_base import AgentBase
from core.utils.data_utils import send_message

logger = logging.getLogger(__name__)

class GeopoliticalRiskAgent(AgentBase):
    """
    Agent responsible for assessing Geopolitical Risk.
    Evaluates political stability, trade relations, conflict risks, and contagion.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.data_sources = config.get('data_sources', {})

        # Base connectivity for contagion (Region -> [Connected Regions])
        self.connectivity_map = {
            "US": ["EU", "CN", "Global"],
            "CN": ["US", "EM", "Global"],
            "EU": ["US", "RU", "Middle East", "Global"],
            "RU": ["EU", "Middle East"],
            "Middle East": ["EU", "US", "RU"],
            "EM": ["CN", "US"]
        }

        # Enhanced Supply Chain Dependency Map (Region -> Dependency Weight)
        self.supply_chain_dependency = {
            "US": {"CN": 0.4, "EU": 0.3},
            "EU": {"CN": 0.3, "RU": 0.4}, # Energy dependency on RU
            "CN": {"US": 0.3, "EM": 0.4},
            "EM": {"CN": 0.5, "US": 0.3}
        }

        # Economic weight for global impact
        self.global_weights = {
            "US": 0.25, "CN": 0.18, "EU": 0.17,
            "RU": 0.03, "Middle East": 0.05, "EM": 0.32
        }

    async def execute(self, input_data: Union[List[str], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess geopolitical risks.

        Args:
            input_data:
                - List of region names (legacy mode).
                - Dict mapping region names to factor dictionaries (enhanced mode).
                  e.g. {"US": {"political_stability": 80, "trade_conflict": 60, ...}}

        Returns:
             Dict with risk assessment per region, contagion analysis, and global score.
        """
        logger.info(f"Assessing geopolitical risks...")

        risk_assessments = {}
        regions_list = []
        detailed_data = {}

        # Normalize Input
        if isinstance(input_data, list):
            regions_list = input_data
        elif isinstance(input_data, str):
            regions_list = [input_data]
        elif isinstance(input_data, dict):
            regions_list = list(input_data.keys())
            detailed_data = input_data
        else:
            return {"error": "Invalid input format"}

        # 1. Individual Assessments
        for region in regions_list:
            factors = detailed_data.get(region)
            score = self.calculate_risk_score(region, factors)

            # Apply Event Specific Impacts (e.g. "Sanctions")
            event_impact = self._apply_event_impacts(factors)
            score += event_impact
            score = min(100.0, score)

            key_risks = self.identify_key_risks(region, score, factors)

            risk_assessments[region] = {
                "risk_score": score,
                "key_risks": key_risks,
                "event_impact_adjustment": event_impact,
                "level": "High" if score > 70 else "Medium" if score > 40 else "Low"
            }

        # 2. Contagion Analysis
        contagion_impact = self.calculate_contagion(risk_assessments)

        # 3. Global Aggregate Score (Weighted)
        global_score = 0.0
        total_weight = 0.0

        for region, assessment in risk_assessments.items():
            weight = self.global_weights.get(region, 0.05)
            global_score += assessment['risk_score'] * weight
            total_weight += weight

        if total_weight > 0:
            global_score /= total_weight
        else:
            # Fallback if unknown regions
            scores = [r['risk_score'] for r in risk_assessments.values()]
            global_score = sum(scores) / len(scores) if scores else 50.0

        result = {
            'global_risk_index': float(global_score),
            'global_risk_level': "High" if global_score > 60 else "Medium" if global_score > 35 else "Low",
            'regional_assessments': risk_assessments,
            'contagion_risks': contagion_impact
        }

        # Send risk assessments to message queue (legacy support)
        try:
            message = {'agent': 'geopolitical_risk_agent', 'risk_assessments': result}
            send_message(message)
        except Exception as e:
            logger.warning(f"Failed to send legacy message: {e}")

        return result

    def calculate_risk_score(self, region: str, factors: Optional[Dict[str, float]] = None) -> float:
        """
        Calculates risk score (0-100, Higher is Worse).
        Factors:
        - political_stability (0-100, Higher is Better) -> Invert
        - trade_conflict (0-100, Higher is Worse)
        - social_unrest (0-100, Higher is Worse)
        - institutional_strength (0-100, Higher is Better) -> Invert
        """
        if factors:
            # Weights
            w_stability = 0.30
            w_trade = 0.25
            w_unrest = 0.25
            w_inst = 0.20

            stability = factors.get("political_stability", 50)
            trade = factors.get("trade_conflict", 50)
            unrest = factors.get("social_unrest", 50)
            inst = factors.get("institutional_strength", 50)

            # Formula: (100 - Stability)*w + Trade*w + Unrest*w + (100-Inst)*w
            score = ((100 - stability) * w_stability +
                     (trade) * w_trade +
                     (unrest) * w_unrest +
                     (100 - inst) * w_inst)

            return float(score)

        # Fallback Risk Map (0-100)
        risk_map = {
            "US": 30, "EU": 35, "CN": 55, "RU": 85,
            "EM": 60, "Middle East": 75, "Global": 45
        }
        return float(risk_map.get(region, 50))

    def identify_key_risks(self, region: str, score: float, factors: Optional[Dict[str, float]]) -> List[str]:
        """Identifies key drivers of risk."""
        risks = []

        if factors:
            if factors.get("political_stability", 100) < 40: risks.append("Political Instability")
            if factors.get("trade_conflict", 0) > 60: risks.append("Trade War Risk")
            if factors.get("social_unrest", 0) > 60: risks.append("Social Unrest")
            if factors.get("institutional_strength", 100) < 40: risks.append("Weak Institutions")
        else:
            # Legacy heuristics
            if region in ["US", "CN"]: risks.append("Trade Tensions")
            if region in ["RU", "Middle East"]: risks.append("Armed Conflict")
            if region == "EU": risks.append("Regulatory Fragmentation")
            if region == "EM": risks.append("Currency Volatility")

        if score > 70 and not risks:
            risks.append("General Elevated Risk")

        return risks

    def calculate_contagion(self, assessments: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Analyzes contagion risk using Connectivity and Supply Chain Dependency.
        If a connected region has High Risk (>60), it imposes a 'contagion penalty' on this region.
        """
        contagion_map = {}

        for region in assessments:
            neighbors = self.connectivity_map.get(region, [])
            dependencies = self.supply_chain_dependency.get(region, {})

            incoming_contagion = 0.0
            sources = []

            for neighbor in neighbors:
                if neighbor in assessments:
                    neighbor_score = assessments[neighbor]['risk_score']
                    if neighbor_score > 60:
                        # Base Contagion Intensity: 15% of the excess risk over 60
                        base_factor = 0.15

                        # Boost factor if there is high supply chain dependency
                        dep_weight = dependencies.get(neighbor, 0.0)
                        factor = base_factor + (dep_weight * 0.20) # Up to +20% more impact

                        impact = (neighbor_score - 60) * factor
                        incoming_contagion += impact
                        sources.append(f"{neighbor}({impact:.1f})")

            contagion_map[region] = {
                "incoming_contagion_score": float(incoming_contagion),
                "sources": sources
            }

        return contagion_map

    def _apply_event_impacts(self, factors: Optional[Dict[str, Any]]) -> float:
        """Applies score adjustments based on boolean flags or keywords."""
        if not factors: return 0.0

        impact = 0.0

        # Check for specific flags
        if factors.get("active_conflict", False): impact += 25
        if factors.get("sanctions_imposed", False): impact += 15
        if factors.get("upcoming_election_uncertainty", False): impact += 10
        if factors.get("supply_chain_disruption", False): impact += 10

        return impact

    def assess_geopolitical_risks(self) -> Dict[str, Any]:
        """Legacy wrapper."""
        return asyncio.run(self.execute(["Global"]))
