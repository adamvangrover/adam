import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from afos_v3.gateway.zero_trust_gateway import QuarantineSink
from scripts.sre_eval_harness import (
    GENESIS_CHAIN_HASH,
    AsyncProvOBatchLedger,
    KernelRPCEnvelope,
    build_synthetic_wave,
    main,
    percentile,
    run_hardened_dag_wave,
    run_hardened_dag_wave_detailed,
    run_naive_dag_wave,
    verify_prov_o_ledger_chain,
)
from src.adam.core.state_manager import StateManager


def test_kernel_rpc_envelope_from_raw_fast_and_grounding() -> None:
    raw = json.dumps(
        {
            "trace_id": "trc-20261106-0001",
            "agent_id": "agent-node-001",
            "domain": "enterprise",
            "epoch_ms": 1794010000001,
            "layer0_policy_ok": 1,
            "layer1_runtime_ok": 1,
            "payload_vector": [0.1, 0.2, 0.3],
            "source_data_object": "urn:adam:kernel:telemetry:v30.1",
            "confidence_score": 0.97,
        }
    ).encode("utf-8")

    env = KernelRPCEnvelope.from_raw_fast(memoryview(raw))
    assert env.trace_id == "trc-20261106-0001"
    assert env.safety_gate == 1
    assert env.payload_vector == (0.1, 0.2, 0.3)
    assert env.check_grounding() is True

    header = env.build_provenance_header()
    assert header.source_data_object == "urn:adam:kernel:telemetry:v30.1"
    assert len(header.content_hash) == 64
    assert len(header.hash_state()) == 64


def test_percentile_interpolation() -> None:
    assert percentile([], 50) == 0.0
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 0) == 10.0
    assert percentile(vals, 50) == 30.0
    assert percentile(vals, 100) == 50.0
    assert percentile(vals, 25) == 20.0


async def _async_test_batch_ledger_safety_gate_and_quarantine(tmp_path: Path) -> None:
    ledger_path = tmp_path / "hardened.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    sink = QuarantineSink(sink_path=quarantine_path)
    state_mgr = StateManager(redis_url="redis://127.0.0.1:0/0")

    ledger = AsyncProvOBatchLedger(
        path=ledger_path,
        quarantine_sink=sink,
        state_manager=state_mgr,
    )
    await ledger.start()
    try:
        valid_env = KernelRPCEnvelope(
            trace_id="trc-valid-01",
            agent_id="agent-01",
            domain="governance",
            epoch_ms=1794010000000,
            layer0_policy_ok=1,
            layer1_runtime_ok=1,
            payload_vector=(0.5, 0.25),
        )
        digest = await ledger.commit_event(valid_env)
        assert len(digest) == 64

        tripped_env = KernelRPCEnvelope(
            trace_id="trc-tripped-02",
            agent_id="agent-02",
            domain="governance",
            epoch_ms=1794010000001,
            layer0_policy_ok=1,
            layer1_runtime_ok=0,
            payload_vector=(0.5, 0.25),
        )
        raised = False
        try:
            await ledger.commit_event(tripped_env)
        except PermissionError as exc:
            assert "Safety gate tripped" in str(exc)
            raised = True
        assert raised is True
    finally:
        await ledger.stop()

    assert ledger.committed_count == 1
    assert ledger.quarantined_count == 1
    assert state_mgr.system_status == "LOCKDOWN"
    assert state_mgr.allocations == {"fortress": 0.75, "convex_hunt": 0.00}
    assert len(sink.records) == 1
    assert sink.records[0].failure_category == "SAFETY_GATE_BREACH"

    chain_ok, count, final_hash = verify_prov_o_ledger_chain(ledger_path)
    assert chain_ok is True
    assert count == 1
    assert final_hash != GENESIS_CHAIN_HASH


def test_async_prov_o_batch_ledger_safety_gate_and_quarantine(tmp_path: Path) -> None:
    asyncio.run(_async_test_batch_ledger_safety_gate_and_quarantine(tmp_path))


async def _async_test_sre_eval_harness_end_to_end(tmp_path: Path) -> None:
    raw_messages = build_synthetic_wave(agent_count=24, adversarial_fault_count=0)
    naive_lat = await run_naive_dag_wave(raw_messages, tmp_path / "naive.jsonl")
    hardened_lat = await run_hardened_dag_wave(
        raw_messages, tmp_path / "hardened.jsonl", max_concurrency=8
    )
    assert len(naive_lat) == 24
    assert len(hardened_lat) == 24

    report = await main(
        agent_count=24,
        max_concurrency=8,
        adversarial_fault_count=2,
        emit_stdout=False,
    )
    assert report["agent_count"] == 24
    assert report["prov_o_audit"]["committed_records"] == 24
    assert report["prov_o_audit"]["chain_verified"] is True
    assert report["adversarial_fault_drill"]["quarantined_records"] == 2
    assert report["adversarial_fault_drill"]["surviving_committed_records"] == 22
    assert report["adversarial_fault_drill"]["post_fault_system_status"] == "LOCKDOWN"
    assert report["governance_gate"]["invariants_passed"] is True
    assert report["governance_gate"]["integration_allowed"] is True


def test_sre_eval_harness_end_to_end(tmp_path: Path) -> None:
    asyncio.run(_async_test_sre_eval_harness_end_to_end(tmp_path))


class TestSREEvalHarness(unittest.TestCase):
    def test_envelope_and_grounding(self) -> None:
        test_kernel_rpc_envelope_from_raw_fast_and_grounding()

    def test_percentiles(self) -> None:
        test_percentile_interpolation()

    def test_ledger_and_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            test_async_prov_o_batch_ledger_safety_gate_and_quarantine(Path(td))

    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            test_sre_eval_harness_end_to_end(Path(td))


if __name__ == "__main__":
    unittest.main()
