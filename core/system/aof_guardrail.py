from typing import Any, Callable, Dict, Optional
import functools
import logging
from core.system.hmm_protocol import HMMParser

logger = logging.getLogger(__name__)

class AgenticOversightFramework:
    """
    Implements the Agentic Oversight Framework (AOF) as described in the Agentic Convergence Whitepaper.
    Enforces "Deterministic HITL Triggers" and the "Four Eyes" principle.
    """

    @staticmethod
    def oversight_guardrail(confidence_threshold: float = 0.85):
        """
        Decorator to enforce AOF guardrails on agent execution methods.

        If the agent's internal confidence/conviction score is below the threshold,
        it raises an OversightException that should trigger an HMM workflow.
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(self, *args, **kwargs):
                # 1. Pre-Execution Checks (ARP validation could go here)
                logger.info(f"AOF: Initiating oversight for {func.__name__} in {self.__class__.__name__}")

                # 2. Execute the Agent Logic
                result = await func(self, *args, **kwargs)

                # 3. Post-Execution Validation (Deterministic HITL Trigger)
                # Check if result has a conviction score or similar metric
                score = 0.0
                if hasattr(result, 'conviction_score'):
                    score = result.conviction_score
                elif hasattr(result, 'conviction_level'):
                     # Normalize 1-10 scale to 0-1
                    score = result.conviction_level / 10.0
                elif isinstance(result, dict) and 'conviction_score' in result:
                    score = result['conviction_score']

                # Force Trigger
                if score < confidence_threshold:
                    logger.warning(f"AOF Trigger: Confidence {score:.2f} < Threshold {confidence_threshold}. Halting for HMM Intervention.")
                    raise OversightInterventionRequired(
                        agent_name=self.__class__.__name__,
                        reason=f"Confidence Score ({score:.2f}) below threshold ({confidence_threshold})",
                        context=kwargs
                    )

                logger.info("AOF: Oversight check passed.")
                return result
            return wrapper
        return decorator

class OversightInterventionRequired(Exception):
    """
    Exception raised when an agent triggers a deterministic HITL rule.
    """
    def __init__(self, agent_name: str, reason: str, context: Dict[str, Any]):
        self.agent_name = agent_name
        self.reason = reason
        self.context = context
        self.hmm_request = self._generate_hmm_request()
        super().__init__(f"Oversight Intervention Required: {reason}")

    def _generate_hmm_request(self) -> str:
        return HMMParser.generate_request(
            action="REVIEW_LOW_CONFIDENCE",
            target=self.agent_name,
            justification=self.reason,
            parameters=self.context
        )
