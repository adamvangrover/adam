import pytest
import asyncio
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from src.backend.temporal.workflows.agent_workflow import AgentExecutionWorkflow
from src.backend.temporal.activities.agent_activities import execute_agent_inference

@pytest.mark.asyncio
async def test_agent_execution_workflow():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="agent-task-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[execute_agent_inference],
        ):
            # Execute the workflow
            result = await env.client.execute_workflow(
                AgentExecutionWorkflow.run,
                args=["underwriter", {"ebitda_margin": 0.20}, {"retrieved_docs": []}],
                id="test-workflow-1",
                task_queue="agent-task-queue",
            )

            assert result["decision"] == "approved"
            assert result["confidence"] == 0.92
            assert "processed_at" in result
