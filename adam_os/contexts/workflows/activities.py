import structlog
from temporalio import activity
from typing import Dict, Any

from adam_os.contexts.governance.engine import DeterministicPolicyEngine, FinancialContext, PolicyResult
from adam_os.contexts.ledger.aggregate import FinancialEntity
from adam_os.core.events import CovenantEvaluated, CovenantBreachedEvent

logger = structlog.get_logger(__name__)

# Sample hardcoded rules for demo
RULES = {
    "softbank_arm_margin_loop": {
        "threshold": 0.35,
        "logic": {"<=": [{"var": "ltv"}, 0.35]}
    }
}

@activity.defn
async def evaluate_covenant_activity(context_data: Dict[str, Any]) -> PolicyResult:
    """Temporal activity to evaluate a covenant deterministically."""
    context = FinancialContext(**context_data)
    engine = DeterministicPolicyEngine(RULES)
    result = engine.evaluate_covenant("softbank_arm_margin_loop", context)

    logger.info(
        "activity_evaluate_covenant",
        entity_id=context.entity_id,
        passed=result.passed,
        ltv=result.evaluated_value
    )
    return result

@activity.defn
async def emit_covenant_breach_event_activity(result_data: Dict[str, Any], entity_id: str) -> None:
    """Temporal activity to emit a breach event to the ledger."""
    # Here it would interact with EventStore/Postgres
    event = CovenantBreachedEvent(
        entity_id=entity_id,
        covenant_name=result_data["covenant_name"],
        evaluated_value=result_data["evaluated_value"],
        threshold_value=result_data["threshold_value"]
    )
    logger.warn("emitted_covenant_breach_event", event_id=event.event_id, entity_id=entity_id)
