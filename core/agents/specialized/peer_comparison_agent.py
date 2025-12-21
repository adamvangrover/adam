import logging
from typing import Any, Dict

from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import MultiplesAnalysis

logger = logging.getLogger(__name__)

class PeerComparisonAgent(AgentBase):
    """
    Phase 2 Helper: Peer Comparison.
    Fetches and calculates relative multiples.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Equity Strategist"

    async def execute(self, **kwargs) -> MultiplesAnalysis:
        logger.info("Executing Peer Comparison...")

        # Support both nested 'params' key and flat kwargs
        params = kwargs.get('params', kwargs)

        company_id = params.get("company_id", "Unknown")

        # Mock Data
        # In production, this fetches live peer data from Bloomberg/FactSet/API.

        return MultiplesAnalysis(
            current_ev_ebitda=14.5,
            peer_median_ev_ebitda=12.0,
            premium_discount="20% Premium"
        )
