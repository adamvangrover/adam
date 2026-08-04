import pytest
import asyncio
from src.backend.orchestration.agent_runtime import AgentOrchestrator, AgentTask

@pytest.mark.asyncio
async def test_orchestrator_successful_execution():
    orchestrator = AgentOrchestrator(memory_client=None, rules_path="rules")
    task = AgentTask(target_agent="underwriter", payload={"ebitda_margin": 0.20, "leverage_ratio": 5.0}, context_keys=["q3_earnings"])

    result = await orchestrator.execute_task(task)

    assert result.status == "SUCCESS"
    assert len(result.prov_o_trail) == 3
    assert result.prov_o_trail[0]["prov:Activity"] == "TaskInitialization"
