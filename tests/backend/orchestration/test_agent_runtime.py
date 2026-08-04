import asyncio
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from src.backend.orchestration.agent_runtime import AgentOrchestrator, AgentTask
from src.backend.temporal.workflows.agent_workflow import AgentExecutionWorkflow
from src.backend.temporal.activities.agent_activities import execute_agent_inference

@pytest.mark.asyncio
async def test_orchestrator_successful_execution_stubbed():
    # Using memory_client as established in the merged Orchestrator
    orchestrator = AgentOrchestrator(memory_client=None, rules_path="rules")
    task = AgentTask(
        target_agent="underwriter", 
        payload={"ebitda_margin": 0.20, "leverage_ratio": 5.0}, 
        context_keys=["q3_earnings"]
    )

    result = await orchestrator.execute_task(task)

    assert result.status == "SUCCESS"
    assert len(result.prov_o_trail) == 3
    assert result.prov_o_trail[0]["prov:Activity"] == "TaskInitialization"

@pytest.mark.asyncio
async def test_orchestrator_successful_execution_temporal():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="agent-task-queue",
            workflows=[AgentExecutionWorkflow],
            activities=[execute_agent_inference],
        ):
            orchestrator = AgentOrchestrator(
                memory_client=None, 
                rules_path="rules", 
                temporal_client=env.client
            )
            task = AgentTask(
                target_agent="underwriter", 
                payload={"ebitda_margin": 0.20, "leverage_ratio": 5.0}, 
                context_keys=["q3_earnings"]
            )

            result = await orchestrator.execute_task(task)

            assert result.status == "SUCCESS"
            assert result.output["decision"] == "approved"
            assert result.output["confidence"] == 0.92
            assert "processed_at" in result.output