from __future__ import annotations
from typing import Any, Dict
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import MultiplesAnalysis

class PeerComparisonAgent(AgentBase):
    async def execute(self, data: Dict[str, Any]) -> MultiplesAnalysis:
        return MultiplesAnalysis(
            current_ev_ebitda=10.0,
            peer_median_ev_ebitda=12.0
        )
