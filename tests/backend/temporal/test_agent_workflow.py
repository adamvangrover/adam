import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from src.backend.temporal.workflows.agent_workflow import AgentExecutionWorkflow
from src.backend.temporal.activities.agent_activities import execute_agent_task_activity
import uuid

@pytest.mark.asyncio
async def test_agent_workflow_execution():
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue="agent-task-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[execute_agent_task_activity],
        ):
            task_dict = {
                "task_id": str(uuid.uuid4()),
                "target_agent": "underwriter",
                "payload": {"ebitda_margin": 0.20, "leverage_ratio": 5.0},
                "context_keys": ["q3_earnings"]
            }

            result = await env.client.execute_workflow(
                AgentExecutionWorkflow.run,
                task_dict,
                id="test-agent-workflow",
                task_queue="agent-task-queue",
            )

            assert result["status"] == "SUCCESS"
