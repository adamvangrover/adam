import os
import uuid
import asyncio
import pytest
import pytest_asyncio
from typing import Any, Dict

# Assuming the engine is located at src.orchestrator.engine
from src.orchestrator.engine import (
    OrchestratorEngine, 
    TaskNode, 
    TaskState, 
    TokenOverflowException, 
    CircularDependencyError
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest_asyncio.fixture
async def engine():
    """
    Provides a fresh OrchestratorEngine with an isolated temporary 
    SQLite database for each test to prevent race conditions.
    """
    db_path = f"test_ledger_{uuid.uuid4().hex[:8]}.db"
    workflow_id = f"wf-test-{uuid.uuid4().hex[:8]}"
    
    eng = OrchestratorEngine(workflow_id)
    eng.db_manager.db_path = db_path
    
    yield eng
    
    # Teardown: Ensure DB worker is stopped and file is cleaned up
    await eng.db_manager.stop()
    if os.path.exists(db_path):
        # Allow a brief moment for aiosqlite file locks to release
        await asyncio.sleep(0.1) 
        os.remove(db_path)


# -------------------------------------------------------------------
# Mock Coroutines
# -------------------------------------------------------------------

async def mock_success(ctx: Dict[str, Any]):
    return {"updates": {"success": True}, "tokens_used": 10}

async def mock_fail(ctx: Dict[str, Any]):
    raise ValueError("Intentional test failure")

async def mock_timeout(ctx: Dict[str, Any]):
    await asyncio.sleep(2.0)
    return {"updates": {}, "tokens_used": 5}


# -------------------------------------------------------------------
# Test Suite
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequential_execution(engine: OrchestratorEngine):
    async def task_1(ctx):
        return {"updates": {"t1": "done"}, "tokens_used": 10}

    async def task_2(ctx):
        assert ctx.get("t1") == "done"
        return {"updates": {"t2": "done"}, "tokens_used": 20}

    engine.add_task(TaskNode("t1", task_1))
    engine.add_task(TaskNode("t2", task_2, dependencies=["t1"]))

    await engine.run()

    assert engine.ledger.status == TaskState.COMPLETED
    assert engine.ledger.global_context == {"t1": "done", "t2": "done"}
    assert engine.ledger.task_registry["t1"]["status"] == TaskState.COMPLETED
    assert engine.ledger.task_registry["t2"]["status"] == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_parallel_execution(engine: OrchestratorEngine):
    async def t_start(ctx): return {"updates": {"start": True}, "tokens_used": 5}
    async def t_a(ctx): 
        await asyncio.sleep(0.1)
        return {"updates": {"a": True}, "tokens_used": 10}
    async def t_b(ctx): 
        await asyncio.sleep(0.1)
        return {"updates": {"b": True}, "tokens_used": 15}
    async def t_end(ctx):
        assert ctx.get("a") and ctx.get("b")
        return {"updates": {"end": True}, "tokens_used": 5}

    engine.add_task(TaskNode("start", t_start))
    engine.add_task(TaskNode("a", t_a, dependencies=["start"]))
    engine.add_task(TaskNode("b", t_b, dependencies=["start"]))
    engine.add_task(TaskNode("end", t_end, dependencies=["a", "b"]))

    await engine.run(max_concurrency=5)

    assert engine.ledger.status == TaskState.COMPLETED
    assert engine.ledger.task_registry["a"]["status"] == TaskState.COMPLETED
    assert engine.ledger.task_registry["b"]["status"] == TaskState.COMPLETED
    assert engine.ledger.task_registry["end"]["status"] == TaskState.COMPLETED


@pytest.mark.asyncio
async def test_skip_on_failed_dependency(engine: OrchestratorEngine):
    engine.add_task(TaskNode("t1", mock_fail))
    engine.add_task(TaskNode("t2", mock_success, dependencies=["t1"]))

    await engine.run()
    
    assert engine.ledger.status == TaskState.FAILED
    assert engine.ledger.task_registry["t1"]["status"] == TaskState.FAILED
    assert engine.ledger.task_registry["t2"]["status"] == TaskState.SKIPPED


@pytest.mark.asyncio
async def test_token_overflow(engine: OrchestratorEngine):
    async def t_heavy(ctx):
        return {"tokens_used": 2000}

    engine.add_task(TaskNode("heavy", t_heavy, max_tokens=1000))
    engine.add_task(TaskNode("next", mock_success, dependencies=["heavy"]))

    await engine.run()

    assert engine.ledger.status == TaskState.FAILED
    assert engine.ledger.task_registry["heavy"]["status"] == TaskState.FAILED
    assert engine.ledger.task_registry["next"]["status"] == TaskState.SKIPPED


@pytest.mark.asyncio
async def test_timeout_and_exponential_backoff_retry(engine: OrchestratorEngine):
    attempt_count = 0
    
    async def flaky_task(ctx):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Temporary network glitch")
        return {"updates": {"recovered": True}, "tokens_used": 10}
        
    engine.add_task(TaskNode(
        "flaky", 
        flaky_task, 
        max_retries=3, 
        retry_backoff=0.1  # Fast backoff for testing
    ))
    
    await engine.run()
    
    assert engine.ledger.status == TaskState.COMPLETED
    assert engine.ledger.task_registry["flaky"]["status"] == TaskState.COMPLETED
    assert attempt_count == 3
    assert engine.ledger.global_context.get("recovered") is True


@pytest.mark.asyncio
async def test_conditional_router_skip(engine: OrchestratorEngine):
    def skip_if_true(context):
        return not context.get("should_skip", False)

    async def set_skip(ctx):
        return {"updates": {"should_skip": True}, "tokens_used": 5}

    engine.add_task(TaskNode("t1", set_skip))
    engine.add_task(TaskNode("t2", mock_success, conditional_router=skip_if_true, dependencies=["t1"]))
    engine.add_task(TaskNode("t3", mock_success, dependencies=["t2"]))

    await engine.run()
    
    assert engine.ledger.status == TaskState.COMPLETED
    assert engine.ledger.task_registry["t1"]["status"] == TaskState.COMPLETED
    # t2 should be skipped due to router, causing t3 to skip due to dependency
    assert engine.ledger.task_registry["t2"]["status"] == TaskState.SKIPPED
    assert engine.ledger.task_registry["t3"]["status"] == TaskState.SKIPPED


@pytest.mark.asyncio
async def test_circular_dependency_detection(engine: OrchestratorEngine):
    engine.add_task(TaskNode("t1", mock_success, dependencies=["t3"]))
    engine.add_task(TaskNode("t2", mock_success, dependencies=["t1"]))
    engine.add_task(TaskNode("t3", mock_success, dependencies=["t2"]))
    
    with pytest.raises(CircularDependencyError):
        await engine.run()


@pytest.mark.asyncio
async def test_human_gate_suspension_and_recovery(engine: OrchestratorEngine):
    async def t_pre(ctx): return {"updates": {"pre": True}, "tokens_used": 10}
    async def t_gate(ctx): return "SUSPENDED_AWAITING_INPUT"
    async def t_post(ctx):
        assert ctx.get("human_approved") is True
        return {"updates": {"post": True}, "tokens_used": 10}

    engine.add_task(TaskNode("pre", t_pre))
    engine.add_task(TaskNode("gate", t_gate, dependencies=["pre"], is_human_gate=True))
    engine.add_task(TaskNode("post", t_post, dependencies=["gate"]))

    await engine.run()

    # Verify suspension
    assert engine.ledger.status == TaskState.SUSPENDED_AWAITING_INPUT
    assert engine.ledger.task_registry["gate"]["status"] == TaskState.SUSPENDED_AWAITING_INPUT

    # Resolve the gate via payload injection
    engine.resolve_human_gate("gate", {
        "updates": {"human_approved": True}, 
        "tokens_used": 5
    })

    # Verify state queued for resume
    assert engine.ledger.task_registry["gate"]["status"] == TaskState.PENDING

    # Run again to finish the DAG
    await engine.run()

    assert engine.ledger.status == TaskState.COMPLETED
    assert engine.ledger.global_context.get("human_approved") is True
    assert engine.ledger.task_registry["post"]["status"] == TaskState.COMPLETED