import logging
from typing import Any, Dict

from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import FinalVerdict, StrategicSynthesis

logger = logging.getLogger(__name__)

class PortfolioManagerAgent(AgentBase):
    """
    Phase 5: Synthesis & Conviction.
    The 'Conviction Engine' that weighs all previous phases.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "CIO"

    async def execute(self, **kwargs) -> StrategicSynthesis:
        logger.info("Executing Strategic Synthesis...")

        # Support both nested 'params' key and flat kwargs
        params = kwargs.get('params', kwargs)

        # In production, this would implement a weighted scoring algorithm
        # based on the outputs of the other agents.

        equity_bullish = True # Simplified
        credit_stable = True

        conviction = 8
        rec = "Long"

        if not credit_stable:
            conviction = 3
            rec = "Hold"

        # Normalize recommendation to new schema
        if rec == "Long": rec = "Buy"
        if rec == "Short": rec = "Sell"
        if rec not in ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]:
             rec = "Hold"

        return StrategicSynthesis(
            m_and_a_posture="Neutral",
            final_verdict=FinalVerdict(
                recommendation=rec,
                conviction_level=conviction,
                time_horizon="12-24 Months",
                rationale_summary="Strong fundamentals and wide moat offset by moderate valuation premiums.",
                justification_trace=[
                    "Management assessment indicates high alignment.",
                    "DCF implies 15% upside.",
                    "Credit risk is contained (Pass rating).",
                    "Monte Carlo shows <2% default probability."
                ]
            )
        )
