from typing import Any, Dict

from core.agents.meta_agents.omega_meta_orchestrator import OmegaMetaOrchestrator
from core.schemas.agent_schema import AgentInput, AgentOutput

class CognitiveHarness:
    """
    Bridge between System 1 real-time swarms and System 2 deep reasoning.
    Evaluates thresholds and triggers System 2 escalation.
    """
    def __init__(self):
        self.orchestrator = OmegaMetaOrchestrator()

    def evaluate_threshold(self, system1_score: float, threshold: float = 0.85) -> bool:
        """
        Evaluates System 1 confidence against the specified threshold.
        Returns True if score is below threshold indicating System 2 intervention is needed.
        """
        return system1_score < threshold

    async def route_to_system2(self, input_dict: Dict[str, Any]) -> AgentOutput:
        """
        Instantiates OmegaMetaOrchestrator and triggers execution.
        """
        query = input_dict.get("query", "System 2 required")
        context = input_dict.get("context", {})

        agent_input = AgentInput(
            query=query,
            context=context,
            tools=[]
        )

        return await self.orchestrator.execute_pydantic(agent_input)
