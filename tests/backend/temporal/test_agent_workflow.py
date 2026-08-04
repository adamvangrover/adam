import uuid
import asyncio
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from src.backend.temporal.workflows.agent_workflow import (
    AgentTaskWorkflow, 
    AgentExecutionWorkflow
)
from src.backend.temporal.activities.agent_activities import (
    execute_agent_task_activity, 
    execute_agent_inference
)


@pytest.mark.asyncio
async def test_agent_task_workflow():
    """
    Tests the high-level orchestration workflow that wraps the AgentOrchestrator logic.
    """
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="agent-task-queue",
            workflows=[AgentTaskWorkflow],
            activities=[execute_agent_task_activity],
        ):
            task_dict = {
                "task_id": str(uuid.uuid4()),
                "target_agent": "underwriter",
                "payload": {"ebitda_margin": 0.20, "leverage_ratio": 5.0},
                "context_keys": ["q3_earnings"]
            }

            result = await env.client.execute_workflow(
                AgentTaskWorkflow.run,
                task_dict,
                id="test-agent-task-workflow",
                task_queue="agent-task-queue",
            )

            assert result["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_agent_execution_workflow():
    """
    Tests the isolated workflow responsible for executing the LLM inference activity.
    """
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="agent-task-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[execute_agent_inference],
        ):
            # Execute the workflow using positional args to match the signature
            result = await env.client.execute_workflow(
                AgentExecutionWorkflow.run,
                args=["underwriter", {"ebitda_margin": 0.20}, {"retrieved_docs": []}],
                id="test-workflow-1",
                task_queue="agent-task-queue",
            )

            assert result["decision"] == "approved"
            assert result["confidence"] == 0.92
            assert "processed_at" in result