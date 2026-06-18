import pytest
import asyncio
from src.orchestrator.engine import StateLedger, TaskNode, OrchestratorEngine, TokenOverflowException

@pytest.mark.asyncio
async def test_sequential_execution():
    ledger = StateLedger("wf-test-seq")
    engine = OrchestratorEngine(ledger)

    async def task_1(ctx):
        return {"updates": {"t1": "done"}, "tokens_used": 10}

    async def task_2(ctx):
        assert ctx.get("t1") == "done"
        return {"updates": {"t2": "done"}, "tokens_used": 20}

    engine.add_task(TaskNode("t1", task_1))
    engine.add_task(TaskNode("t2", task_2, dependencies=["t1"]))

    await engine.run()

    assert ledger.status == "COMPLETED"
    assert ledger.global_context == {"t1": "done", "t2": "done"}
    assert ledger.task_registry["t1"]["status"] == "COMPLETED"
    assert ledger.task_registry["t2"]["status"] == "COMPLETED"

@pytest.mark.asyncio
async def test_parallel_execution():
    ledger = StateLedger("wf-test-par")
    engine = OrchestratorEngine(ledger)

    async def t_start(ctx): return {"updates": {"start": True}, "tokens_used": 5}
    async def t_a(ctx): return {"updates": {"a": True}, "tokens_used": 10}
    async def t_b(ctx): return {"updates": {"b": True}, "tokens_used": 15}
    async def t_end(ctx):
        assert ctx.get("a") and ctx.get("b")
        return {"updates": {"end": True}, "tokens_used": 5}

    engine.add_task(TaskNode("start", t_start))
    engine.add_task(TaskNode("a", t_a, dependencies=["start"]))
    engine.add_task(TaskNode("b", t_b, dependencies=["start"]))
    engine.add_task(TaskNode("end", t_end, dependencies=["a", "b"]))

    await engine.run()

    assert ledger.status == "COMPLETED"
    assert ledger.task_registry["a"]["status"] == "COMPLETED"
    assert ledger.task_registry["b"]["status"] == "COMPLETED"
    assert ledger.task_registry["end"]["status"] == "COMPLETED"

@pytest.mark.asyncio
async def test_token_overflow():
    ledger = StateLedger("wf-test-tok")
    engine = OrchestratorEngine(ledger)

    async def t_heavy(ctx):
        return {"tokens_used": 2000}

    async def t_next(ctx):
        return {"tokens_used": 10}

    engine.add_task(TaskNode("heavy", t_heavy, max_tokens=1000))
    engine.add_task(TaskNode("next", t_next, dependencies=["heavy"]))

    await engine.run()

    assert ledger.status == "FAILED"
    assert ledger.task_registry["heavy"]["status"] == "FAILED"
    assert ledger.task_registry["next"]["status"] == "SKIPPED"

@pytest.mark.asyncio
async def test_human_gate_suspension_and_recovery():
    ledger = StateLedger("wf-test-gate")
    engine = OrchestratorEngine(ledger)

    async def t_pre(ctx): return {"updates": {"pre": True}, "tokens_used": 10}
    async def t_gate(ctx): return "SUSPENDED_AWAITING_INPUT"
    async def t_post(ctx):
        assert ctx.get("human_approved") is True
        return {"updates": {"post": True}, "tokens_used": 10}

    engine.add_task(TaskNode("pre", t_pre))
    engine.add_task(TaskNode("gate", t_gate, dependencies=["pre"], is_human_gate=True))
    engine.add_task(TaskNode("post", t_post, dependencies=["gate"]))

    await engine.run()

    assert ledger.status == "SUSPENDED"
    assert ledger.task_registry["gate"]["status"] == "SUSPENDED"

    # Simulate recovery
    serialized_state = ledger.serialize()

    new_ledger = StateLedger("wf-test-gate")
    new_engine = OrchestratorEngine(new_ledger)
    new_engine.load_checkpoint(serialized_state)

    new_engine.add_task(TaskNode("pre", t_pre))
    new_engine.add_task(TaskNode("gate", t_gate, dependencies=["pre"], is_human_gate=True))
    new_engine.add_task(TaskNode("post", t_post, dependencies=["gate"]))

    new_engine.resolve_human_gate("gate", {"updates": {"human_approved": True}, "tokens_used": 0})

    assert new_ledger.task_registry["gate"]["status"] == "COMPLETED"
    assert new_ledger.status == "PENDING"

    await new_engine.run()

    assert new_ledger.status == "COMPLETED"
    assert new_ledger.global_context.get("human_approved") is True
    assert new_ledger.task_registry["post"]["status"] == "COMPLETED"
