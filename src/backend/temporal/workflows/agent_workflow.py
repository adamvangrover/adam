from datetime import timedelta
from typing import Any, Dict
from temporalio import workflow
from temporalio.common import RetryPolicy

# Import the activities. Note: inside workflows, imports of non-deterministic code
# or standard datetime functions should be avoided or carefully managed.
with workflow.unsafe.imports_passed_through():
    from src.backend.temporal.activities.agent_activities import (
        execute_agent_task_activity,
        execute_agent_inference,
        AgentExecutionParams
    )


@workflow.defn
class AgentTaskWorkflow:
    """
    Deterministic Temporal workflow for orchestrating high-level agent tasks.
    This manages the complete lifecycle wrapper including policy checks and memory injection.
    """
    @workflow.run
    async def run(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        workflow.logger.info("AgentTaskWorkflow started", extra={"task_id": task_dict.get("task_id")})

        # Execute the full agent task as an activity
        result = await workflow.execute_activity(
            execute_agent_task_activity,
            task_dict,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=10)
            )
        )

        workflow.logger.info("AgentTaskWorkflow completed", extra={"status": result.get("status")})
        return result


@workflow.defn
class AgentExecutionWorkflow:
    """
    Deterministic workflow orchestrating isolated agent execution (inference).
    Called by the Orchestrator runtime once rules and memory have been resolved.
    """
    @workflow.run
    async def run(self, agent_name: str, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        workflow.logger.info(f"Starting AgentExecutionWorkflow for agent: {agent_name}")

        params = AgentExecutionParams(
            agent_name=agent_name,
            payload=payload,
            context=context
        )

        # Execute the inference activity with strict retry policies
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

        workflow.logger.info(f"AgentExecutionWorkflow completed for agent: {agent_name}")
        return result