import logging
from typing import Any, Dict, Optional

from core.agents.agent_base import AgentBase
from core.v23_graph_engine.odyssey_knowledge_graph import OdysseyKnowledgeGraph

logger = logging.getLogger(__name__)

class CreditSentryAgent(AgentBase):
    """
    "The Hawk" - Solvency Assessment Engine.
    Responsibility: Stress testing, FCCR calculation, Cycle Detection (Fractured Ouroboros), J.Crew Detection.
    """
    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None, graph: Optional[OdysseyKnowledgeGraph] = None):
        super().__init__(config, kernel=kernel)
        self.graph = graph or OdysseyKnowledgeGraph()

    async def execute(self, **kwargs) -> Dict[str, Any]:
        entity_id = kwargs.get("entity_id")
        stress_scenario = kwargs.get("stress_scenario", {}) # e.g. {"sofr_hike_bps": 500}

        if not entity_id:
            return {"error": "Missing entity_id"}

        logger.info(f"CreditSentry analyzing {entity_id} under scenario {stress_scenario}")

        # 1. Structural Risk Analysis
        j_crew_risk = self.graph.detect_j_crew_maneuver(entity_id)
        ouroboros_cycles = self.graph.detect_fractured_ouroboros() # This is global, but relevant

        # 2. Solvency Analysis (Mocked calculation)
        fccr = self._calculate_stress_fccr(entity_id, stress_scenario)

        status = "Pass"
        if fccr < 1.0:
            status = "Zombie"

        return {
            "entity_id": entity_id,
            "fccr_stress": fccr,
            "status": status,
            "j_crew_risk": j_crew_risk,
            "circular_dependencies_detected": len(ouroboros_cycles) > 0
        }

    def _calculate_stress_fccr(self, entity_id: str, scenario: Dict[str, Any]) -> float:
        # Fetch financials from graph (mocked access)
        # In reality: self.graph.nodes[entity_id]['data']['financials']...
        # Assume base FCCR is 1.2
        base_fccr = 1.2
        hike = scenario.get("sofr_hike_bps", 0)

        # Simple impact model: 100bps hike -> -0.1 FCCR
        impact = (hike / 100.0) * 0.1
        return base_fccr - impact
