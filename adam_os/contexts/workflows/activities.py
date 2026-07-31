from typing import Dict, Any
import structlog
from temporalio import activity
from pydantic import BaseModel

from adam_os.contexts.governance.engine import DeterministicPolicyEngine, PolicyResult
from adam_os.core.events import CovenantEvaluated

logger = structlog.get_logger()

class EvaluateCovenantInput(BaseModel):
    entity_id: str
    rule_name: str
    context: Dict[str, Any]

class FlagAssetInput(BaseModel):
    entity_id: str
    reason: str
    covenant_type: str
    evaluation_details: Dict[str, Any]

@activity.defn
async def evaluate_covenant(input_data: EvaluateCovenantInput) -> PolicyResult:
    """Temporal activity to evaluate a financial covenant using the deterministic policy engine."""
    logger.info("evaluating_covenant", entity_id=input_data.entity_id, rule_name=input_data.rule_name)
    engine = DeterministicPolicyEngine()

    # In a real system, we might fetch the rule definition from a database or config
    # but here the engine has it hardcoded for simplicity as per the requirement
    result = engine.evaluate(input_data.rule_name, input_data.context)

    logger.info("covenant_evaluated", entity_id=input_data.entity_id, is_breached=result.is_breached)
    return result

@activity.defn
async def flag_asset(input_data: FlagAssetInput) -> CovenantEvaluated:
    """Temporal activity to flag an asset, generating a CovenantEvaluated event."""
    logger.info("flagging_asset", entity_id=input_data.entity_id, reason=input_data.reason)

    event = CovenantEvaluated(
        entity_id=input_data.entity_id,
        covenant_type=input_data.covenant_type,
        is_breached=True,
        evaluation_details=input_data.evaluation_details
    )

    # In a real system, this would append to the Event Sourced ledger.
    # For this clean-room build, we just return the event.
    return event
