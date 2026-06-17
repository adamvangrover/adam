import pytest
import asyncio
from typing import Any, Dict
from src.orchestrator.engine import OrchestrationEngine, TaskNode, TaskState

async def mock_task_success(context: Dict[str, Any]):
    context["success"] = True
    return "result", {"metric": 1}

async def mock_task_fail(context: Dict[str, Any]):
    raise ValueError("Test error")

async def mock_task_timeout(context: Dict[str, Any]):
    await asyncio.sleep(1.0)
    return "timeout", {}

@pytest.mark.asyncio
async def test_orchestrator_sequence():
    engine = OrchestrationEngine("wf-test-1", "tr-test-1")
    engine.register_task(TaskNode(id="t1", func=mock_task_success))
    engine.register_task(TaskNode(id="t2", func=mock_task_success, dependencies=["t1"]))

    ledger = await engine.run()
    assert ledger.metadata["status"] == "COMPLETED"
    assert ledger.task_registry["t1"]["status"] == TaskState.COMPLETED
    assert ledger.task_registry["t2"]["status"] == TaskState.COMPLETED
    assert ledger.global_context["success"] is True

@pytest.mark.asyncio
async def test_orchestrator_skip_on_fail():
    engine = OrchestrationEngine("wf-test-2", "tr-test-2")
    engine.register_task(TaskNode(id="t1", func=mock_task_fail))
    engine.register_task(TaskNode(id="t2", func=mock_task_success, dependencies=["t1"]))

    ledger = await engine.run()
    assert ledger.metadata["status"] == "INCOMPLETE"
    assert ledger.task_registry["t1"]["status"] == TaskState.FAILED
    assert ledger.task_registry["t2"]["status"] == TaskState.SKIPPED

@pytest.mark.asyncio
async def test_orchestrator_timeout():
    engine = OrchestrationEngine("wf-test-3", "tr-test-3")
    engine.register_task(TaskNode(id="t1", func=mock_task_timeout, timeout=0.1))

    ledger = await engine.run()
    assert ledger.metadata["status"] == "INCOMPLETE"
    assert ledger.task_registry["t1"]["status"] == TaskState.FAILED

@pytest.mark.asyncio
async def test_orchestrator_human_gate():
    engine = OrchestrationEngine("wf-test-4", "tr-test-4")
    engine.register_task(TaskNode(id="t1", func=mock_task_success, is_human_gate=True))
    engine.register_task(TaskNode(id="t2", func=mock_task_success, dependencies=["t1"]))

    ledger = await engine.run()
    assert ledger.metadata["status"] == "INCOMPLETE"
    assert ledger.task_registry["t1"]["status"] == TaskState.SUSPENDED_AWAITING_INPUT
    assert ledger.task_registry["t2"]["status"] == TaskState.PENDING

    # Receive input
    engine.receive_human_input("t1", {"input": "from human"})

    ledger = await engine.run()
    assert ledger.task_registry["t1"]["status"] == TaskState.COMPLETED
    assert ledger.task_registry["t2"]["status"] == TaskState.COMPLETED

@pytest.mark.asyncio
async def test_orchestrator_checkpointer_resume():
    engine = OrchestrationEngine("wf-test-5", "tr-test-5")

    engine.register_task(TaskNode(id="t1", func=mock_task_success))
    engine.register_task(TaskNode(id="t2", func=mock_task_success, dependencies=["t1"]))
    engine.register_task(TaskNode(id="t3", func=mock_task_success, dependencies=["t2"]))

    # Simulate a ledger that has completed t1 and t2, but t3 failed
    engine.ledger.task_registry = {
        "t1": {"status": TaskState.COMPLETED, "span_id": "sp-t1", "parent_id": None},
        "t2": {"status": TaskState.COMPLETED, "span_id": "sp-t2", "parent_id": "sp-t1"},
        "t3": {"status": TaskState.FAILED, "span_id": "sp-t3", "parent_id": "sp-t2"}
    }

    ledger = await engine.run()
    assert ledger.metadata["status"] == "COMPLETED"
    assert ledger.task_registry["t3"]["status"] == TaskState.COMPLETED

@pytest.mark.asyncio
async def test_orchestrator_conditional_router():
    engine = OrchestrationEngine("wf-test-6", "tr-test-6")

    def skip_if_true(context):
        return not context.get("should_skip", False)

    async def set_skip(context):
        context["should_skip"] = True
        return "set", {}

    engine.register_task(TaskNode(id="t1", func=set_skip))
    engine.register_task(TaskNode(id="t2", func=mock_task_success, conditional_router=skip_if_true, dependencies=["t1"]))
    engine.register_task(TaskNode(id="t3", func=mock_task_success, dependencies=["t2"]))

    ledger = await engine.run()
    assert ledger.metadata["status"] == "COMPLETED"
    assert ledger.task_registry["t1"]["status"] == TaskState.COMPLETED
    assert ledger.task_registry["t2"]["status"] == TaskState.SKIPPED
    assert ledger.task_registry["t3"]["status"] == TaskState.SKIPPED
