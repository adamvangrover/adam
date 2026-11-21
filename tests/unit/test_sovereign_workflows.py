import pytest
import asyncio
from core.v30_architecture.python_intelligence.workflows.sovereign_workflow import SovereignWorkflowEngine, SovereignWorkflowConfig, RunMode
from core.v30_architecture.python_intelligence.agents.sovereign_orchestrator import SovereignOrchestrator

@pytest.mark.asyncio
async def test_autonomous_workflow():
    orchestrator = SovereignOrchestrator()
    config = SovereignWorkflowConfig(mode=RunMode.AUTONOMOUS)
    engine = SovereignWorkflowEngine(orchestrator, config)

    result = await engine.execute_task("Analyze market volatility")
    assert result["status"] == "success"
    assert "result" in result

@pytest.mark.asyncio
async def test_manual_workflow_approved():
    async def mock_human_approve(task):
        return True

    orchestrator = SovereignOrchestrator()
    config = SovereignWorkflowConfig(mode=RunMode.MANUAL, human_approval_callback=mock_human_approve)
    engine = SovereignWorkflowEngine(orchestrator, config)

    result = await engine.execute_task("Execute high risk trade")
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_manual_workflow_denied():
    async def mock_human_deny(task):
        return False

    orchestrator = SovereignOrchestrator()
    config = SovereignWorkflowConfig(mode=RunMode.MANUAL, human_approval_callback=mock_human_deny)
    engine = SovereignWorkflowEngine(orchestrator, config)

    result = await engine.execute_task("Execute high risk trade")
    assert result["status"] == "rejected"
    assert result["reason"] == "Human denied execution."

@pytest.mark.asyncio
async def test_supervised_workflow_whitelist():
    orchestrator = SovereignOrchestrator()
    config = SovereignWorkflowConfig(mode=RunMode.SUPERVISED, whitelist=["Safe Task"])
    engine = SovereignWorkflowEngine(orchestrator, config)

    # Whitelisted
    result = await engine.execute_task("Safe Task")
    assert result["status"] == "success"

    # Not whitelisted, no callback
    result2 = await engine.execute_task("Unsafe Task")
    assert result2["status"] == "rejected"

@pytest.mark.asyncio
async def test_supervised_workflow_fallback_callback():
    async def mock_human_approve(task):
        return True

    orchestrator = SovereignOrchestrator()
    config = SovereignWorkflowConfig(mode=RunMode.SUPERVISED, whitelist=["Safe Task"], human_approval_callback=mock_human_approve)
    engine = SovereignWorkflowEngine(orchestrator, config)

    # Not whitelisted, but human approves
    result = await engine.execute_task("Unsafe Task")
    assert result["status"] == "success"
