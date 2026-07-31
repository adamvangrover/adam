from datetime import timedelta
import structlog
from temporalio import workflow

# Import activities, fallback for local testing if needed
with workflow.unsafe.imports_passed_through():
    from adam_os.contexts.workflows.activities import (
        evaluate_covenant,
        flag_asset,
        EvaluateCovenantInput,
        FlagAssetInput
    )

logger = structlog.get_logger()

@workflow.defn
class PortfolioSurveillanceWorkflow:
    def __init__(self) -> None:
        self.is_completed = False
        self._asset_state = None
        self._signal_received = False
        self.breach_handled = False

    @workflow.signal
    async def reevaluate_asset(self, arg: dict) -> None:
        """Signal to re-evaluate an asset's state."""
        self._asset_state = {"debt": arg.get("debt", 0.0), "asset_value": arg.get("asset_value", 0.0)}
        self._signal_received = True

    @workflow.run
    async def run(self, entity_id: str) -> None:
        """Main workflow execution loop."""
        workflow.logger.info(f"started_surveillance_workflow entity_id={entity_id}")

        # Wait for a signal to evaluate the asset
        await workflow.wait_condition(lambda: self._signal_received)

        # Reset signal for subsequent runs if this were a long-running loop
        self._signal_received = False

        # Prepare context for the policy engine
        context = {
            "debt": self._asset_state["debt"],
            "asset_value": self._asset_state["asset_value"]
        }

        # Evaluate the 35% LTV covenant using the activity
        policy_result = await workflow.execute_activity(
            evaluate_covenant,
            EvaluateCovenantInput(
                entity_id=entity_id,
                rule_name="ltv_35",
                context=context
            ),
            start_to_close_timeout=timedelta(seconds=10)
        )

        # If breached, emit a command to flag the asset via activity
        if policy_result.is_breached:
            await workflow.execute_activity(
                flag_asset,
                FlagAssetInput(
                    entity_id=entity_id,
                    reason="LTV > 35%",
                    covenant_type="ltv_35",
                    evaluation_details=policy_result.evaluation_details
                ),
                start_to_close_timeout=timedelta(seconds=10)
            )
            self.breach_handled = True

        self.is_completed = True
        workflow.logger.info(f"completed_surveillance_workflow entity_id={entity_id}")
