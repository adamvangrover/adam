from datetime import timedelta
from typing import Dict, Any, List, Optional
from temporalio import workflow

# Import activities, fallback for local testing if needed
with workflow.unsafe.imports_passed_through():
    from adam_os.contexts.workflows.activities import (
        evaluate_covenant,
        flag_asset,
        EvaluateCovenantInput,
        FlagAssetInput
    )

@workflow.defn
class PortfolioSurveillanceWorkflow:
    """
    Long-running Temporal workflow that acts as a continuous surveillance monitor
    for a specific financial entity. Stays active to process asynchronous signals.
    """
    def __init__(self) -> None:
        self._is_active: bool = True
        self._pending_updates: List[Dict[str, Any]] = []
        self.breach_history: List[str] = []

    @workflow.signal
    async def reevaluate_asset(self, context_update: Dict[str, Any]) -> None:
        """
        Signal to queue an asset's state for evaluation. 
        Continuously accessible: Can be called multiple times over the entity's lifecycle.
        """
        self._pending_updates.append(context_update)

    @workflow.signal
    async def terminate_surveillance(self) -> None:
        """Gracefully shutdown the surveillance loop."""
        self._is_active = False

    @workflow.query
    def get_breach_history(self) -> List[str]:
        """Query to inspect the active memory of breaches."""
        return self.breach_history

    @workflow.run
    async def run(self, entity_id: str, custom_ruleset: Optional[Dict[str, Any]] = None) -> None:
        """Main workflow execution loop."""
        workflow.logger.info(f"started_continuous_surveillance_workflow entity_id={entity_id}")

        while self._is_active:
            # Wait for a signal to arrive or termination
            await workflow.wait_condition(
                lambda: len(self._pending_updates) > 0 or not self._is_active
            )

            # Drain the queue of pending updates
            while self._pending_updates:
                current_context = self._pending_updates.pop(0)
                rule_to_evaluate = current_context.pop("target_rule", "softbank_arm_margin_loop")

                # Evaluate the covenant using the merged activity schema
                policy_result = await workflow.execute_activity(
                    evaluate_covenant,
                    EvaluateCovenantInput(
                        entity_id=entity_id,
                        rule_name=rule_to_evaluate,
                        context=current_context,
                        custom_ruleset=custom_ruleset
                    ),
                    start_to_close_timeout=timedelta(seconds=10)
                )

                # If breached, emit a command to flag the asset via activity
                if policy_result.is_breached or policy_result.alert_triggered:
                    await workflow.execute_activity(
                        flag_asset,
                        FlagAssetInput(
                            entity_id=entity_id,
                            reason=f"System Alert: Breach of {policy_result.covenant_name}",
                            covenant_name=policy_result.covenant_name,
                            evaluated_value=policy_result.evaluated_value,
                            threshold_value=policy_result.threshold_value,
                            evaluation_details=policy_result.evaluation_details
                        ),
                        start_to_close_timeout=timedelta(seconds=10)
                    )
                    
                    # Store state in workflow memory for immediate queryability
                    self.breach_history.append(policy_result.covenant_name)
                    workflow.logger.warn(f"covenant_breach_handled entity_id={entity_id} rule={policy_result.covenant_name}")

        workflow.logger.info(f"completed_surveillance_workflow entity_id={entity_id}")