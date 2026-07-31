from datetime import timedelta
from typing import Any, Dict
from temporalio import workflow

# Import the activity. Note: inside workflows, imports of non-deterministic code
# or standard datetime functions should be avoided or carefully managed.
with workflow.unsafe.imports_passed_through():
    from src.backend.temporal.activities.agent_activities import execute_agent_inference, AgentExecutionParams

@workflow.defn
class AgentExecutionWorkflow:
    """
    Deterministic workflow orchestrating agent execution.
    """
    @workflow.run
    async def run(self, agent_name: str, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        workflow.logger.info(f"Starting workflow for agent: {agent_name}")

        params = AgentExecutionParams(
            agent_name=agent_name,
            payload=payload,
            context=context
        )

        # Execute the activity with strict retry policies
        from temporalio.common import RetryPolicy
        result = await workflow.execute_activity(
            execute_agent_inference,
            params,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
            )
        )

        workflow.logger.info(f"Workflow completed for agent: {agent_name}")
        return result
