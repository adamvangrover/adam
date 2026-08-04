from datetime import timedelta
from typing import Dict, Any
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.backend.temporal.activities.agent_activities import execute_agent_task_activity

@workflow.defn
class AgentExecutionWorkflow:
    """
    Deterministic Temporal workflow for orchestrating agents.
    """
    @workflow.run
    async def run(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        workflow.logger.info("AgentExecutionWorkflow started", extra={"task_id": task_dict.get("task_id")})

        # Execute the agent task as an activity
        result = await workflow.execute_activity(
            execute_agent_task_activity,
            task_dict,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=10)
            )
        )

        workflow.logger.info("AgentExecutionWorkflow completed", extra={"status": result.get("status")})
        return result
