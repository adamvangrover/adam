import logging
from typing import Dict, Any, Optional
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)


class OdysseyMetaAgent(AgentBase):
    """
    Strategic Synthesis Agent.
    Aggregates inputs from Sentinel, CreditSentry, Argus, etc. to produce final XML decision.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        super().__init__(config, kernel=kernel)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        # Expects results from other agents
        sentry_result = kwargs.get("credit_sentry_result", {})
        sentinel_result = kwargs.get("sentinel_result", {})

        logger.info("Odyssey synthesizing credit decision...")

        recommendation = "HOLD"
        confidence = 0.8

        if sentry_result.get("status") == "Zombie":
            recommendation = "SELL/EXIT"
            confidence = 0.95

        if sentinel_result.get("flag") == "FLAG_DATA_MISSING":
            recommendation = "HALT_TRADING"
            confidence = 1.0

        # XML Output as requested
        decision_xml = f"""
        <CreditDecision>
            <Recommendation>{recommendation}</Recommendation>
            <Confidence>{confidence}</Confidence>
            <Rationale>
                SentryStatus: {sentry_result.get('status', 'Unknown')}
                DataFlags: {sentinel_result.get('flag', 'None')}
            </Rationale>
        </CreditDecision>
        """

        return {
            "decision_xml": decision_xml.strip(),
            "rationale": f"Based on Sentry status: {sentry_result.get('status')} and Sentinel flags: {sentinel_result.get('flag')}"
        }
