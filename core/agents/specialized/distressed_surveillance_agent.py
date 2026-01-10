import logging
from typing import Dict, Any, Optional
from core.agents.agent_base import AgentBase
from core.engine.surveillance_graph import surveillance_graph_app
from core.engine.states import init_surveillance_state

logger = logging.getLogger(__name__)

class DistressedSurveillanceAgent(AgentBase):
    """
    Agent responsible for identifying 'Zombie Issuers' in the BSL market.
    Wraps the SurveillanceGraph.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        super().__init__(config, kernel=kernel)
        self.graph = surveillance_graph_app

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Executes the surveillance workflow.
        Expected kwargs: sector (str)
        """
        sector = kwargs.get("sector", "General")
        logger.info(f"DistressedSurveillanceAgent executing for sector: {sector}")

        initial_state = init_surveillance_state(sector=sector)
        config = {"configurable": {"thread_id": "surveillance_1"}}

        try:
            if hasattr(self.graph, 'ainvoke'):
                result = await self.graph.ainvoke(initial_state, config=config)
            else:
                result = self.graph.invoke(initial_state, config=config)

            return {
                "status": "success",
                "watchlist": result.get("watchlist", []),
                "report": result.get("final_report", "")
            }
        except Exception as e:
            logger.error(f"Surveillance execution failed: {e}")
            return {"status": "error", "error": str(e)}
