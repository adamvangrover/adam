from datetime import timedelta
from typing import Dict, Any
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from adam_os.contexts.workflows.activities import evaluate_covenant_activity, emit_covenant_breach_event_activity

@workflow.defn
class PortfolioSurveillanceWorkflow:
    @workflow.run
    async def run(self, context_data: Dict[str, Any]) -> None:
        # Determine if asset breaches covenant using jsonLogic activity
        result = await workflow.execute_activity(
            evaluate_covenant_activity,
            context_data,
            start_to_close_timeout=timedelta(seconds=10),
        )

        # If breached, emit event to ledger
        if result.alert_triggered:
            await workflow.execute_activity(
                emit_covenant_breach_event_activity,
                args=[result.model_dump(), context_data["entity_id"]],
                start_to_close_timeout=timedelta(seconds=10),
            )
