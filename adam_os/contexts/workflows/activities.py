from typing import Dict, Any, Optional
import structlog
from temporalio import activity
from pydantic import BaseModel, Field

from adam_os.contexts.governance.engine import DeterministicPolicyEngine, PolicyResult
from adam_os.core.events import CovenantEvaluated, CovenantBreachedEvent

logger = structlog.get_logger(__name__)

# Sample hardcoded rules for fallback, preserving main's specific logic
DEFAULT_RULES = {
    "softbank_arm_margin_loop": {
        "threshold": 0.35,
        "logic": {"<=": [{"var": "ltv"}, 0.35]}
    }
}

class EvaluateCovenantInput(BaseModel):
    """Input schema for covenant evaluation activity."""
    entity_id: str
    rule_name: str
    context: Dict[str, Any]
    # Allow workflows to inject dynamic rulesets at runtime for continuous expandability
    custom_ruleset: Optional[Dict[str, Any]] = Field(default=None, description="Injectable ruleset to override defaults")

class FlagAssetInput(BaseModel):
    """Merged input schema for flagging an asset and emitting breach events."""
    entity_id: str
    reason: str
    covenant_name: str
    evaluated_value: Optional[float] = None
    threshold_value: Optional[float] = None
    evaluation_details: Dict[str, Any] = Field(default_factory=dict)

@activity.defn
async def evaluate_covenant(input_data: EvaluateCovenantInput) -> PolicyResult:
    """
    Temporal activity to evaluate a financial covenant.
    Dynamically loads rules and executes deterministically.
    """
    logger.info("evaluating_covenant", entity_id=input_data.entity_id, rule_name=input_data.rule_name)
    
    # Initialize engine with custom rules if provided, otherwise fallback to defaults
    rules = input_data.custom_ruleset or DEFAULT_RULES
    engine = DeterministicPolicyEngine(rules)

    result = engine.evaluate(input_data.rule_name, input_data.context)

    logger.info(
        "activity_evaluate_covenant_complete", 
        entity_id=input_data.entity_id, 
        rule_name=input_data.rule_name,
        passed=result.passed,
        is_breached=result.is_breached,
        evaluated_value=result.evaluated_value
    )
    
    return result

@activity.defn
async def flag_asset(input_data: FlagAssetInput) -> Dict[str, Any]:
    """
    Temporal activity to flag an asset and emit ledger events.
    Combines both evaluation audit details (HEAD) and strict threshold metrics (main).
    """
    logger.info("flagging_asset", entity_id=input_data.entity_id, reason=input_data.reason)

    # HEAD: Generic evaluation audit event
    evaluated_event = CovenantEvaluated(
        entity_id=input_data.entity_id,
        covenant_type=input_data.covenant_name,
        is_breached=True,
        evaluation_details=input_data.evaluation_details
    )

    # Main: Strict metrics breach event
    breach_event = CovenantBreachedEvent(
        entity_id=input_data.entity_id,
        covenant_name=input_data.covenant_name,
        evaluated_value=input_data.evaluated_value,
        threshold_value=input_data.threshold_value
    )

    logger.warn(
        "emitted_covenant_events", 
        entity_id=input_data.entity_id, 
        covenant=input_data.covenant_name,
        reason=input_data.reason
    )

    # In a real system, these append to the Event Sourced ledger.
    # For this clean-room build, we return both for the Temporal workflow to process.
    return {
        "evaluated_event": evaluated_event,
        "breach_event": breach_event
    }