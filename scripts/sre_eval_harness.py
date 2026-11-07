#!/usr/bin/env python3
"""
sre_eval_harness.py
Deterministic SRE Latency, Throughput & Concurrency Benchmark Harness for adamvangrover/adam (/core_kernel).

Compares Baseline (Un-throttled DAG fan-out + double JSON deserialization + synchronous PROV-O I/O)
against Hardened (Bounded Semaphore + Memoryview/Zero-Copy Validation + Async Batch W3C PROV-O Ledger
+ Monotonic SHA-256 Hash Chaining + Layer0×Layer1 Multiplicative Gate + QuarantineSink & LOCKDOWN Circuit Breaker).

Integrates directly with:
  - `src.pdil.models.ProvenanceHeader` & `src.schemas.core_types.compute_deterministic_hash`
  - `afos_v3.gateway.zero_trust_gateway.QuarantineSink`
  - `src.adam.core.state_manager.StateManager` (75% Fortress / 25% Convex Hunt Barbell Circuit Breaker)
  - `core.evaluation.invariant_enforcer.enforce_invariants` & `core.evaluation.integration_verifier.decide_integration`
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root is importable when executed as a standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afos_v3.gateway.zero_trust_gateway import QuarantineRecord, QuarantineSink
from core.evaluation.integration_verifier import IntegrationDecision, decide_integration
from core.evaluation.invariant_enforcer import GateResult, enforce_invariants
from src.adam.core.state_manager import StateManager

try:
    from src.pdil.models import ProvenanceHeader
    from src.schemas.core_types import compute_deterministic_hash
except ImportError:
    def compute_deterministic_hash(data: dict[str, Any]) -> str:
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @dataclass(frozen=True, slots=True)
    class ProvenanceHeader:  # type: ignore[no-redef]
        git_commit_hash: str
        timestamp: str
        content_hash: str
        jsonLogic_version: str
        confidence_score: float
        derivation_path: str
        source_data_object: str

        def hash_state(self) -> str:
            data = f"{self.git_commit_hash}:{self.jsonLogic_version}".encode("utf-8")
            return hashlib.sha256(data).hexdigest()


GENESIS_CHAIN_HASH: str = "0" * 64


@dataclass(frozen=True, slots=True)
class KernelRPCEnvelope:
    """
    Immutable, slot-backed RPC envelope for `/core_kernel` DAG nodes.
    Supports zero-copy `memoryview` / `bytes` deserialization and direct W3C PROV-O header binding.
    """

    trace_id: str
    agent_id: str
    domain: str
    epoch_ms: int
    layer0_policy_ok: int
    layer1_runtime_ok: int
    payload_vector: tuple[float, ...]
    source_data_object: str = "urn:adam:kernel:telemetry:v30.1"
    confidence_score: float = 0.95
    git_commit_hash: str = "v30.1-afos-kernel"
    json_logic_version: str = "1.0.0"

    @property
    def safety_gate(self) -> int:
        """Multiplicative safety gate: Layer 0 (Policy) x Layer 1 (Runtime)."""
        return int(self.layer0_policy_ok) * int(self.layer1_runtime_ok)

    @classmethod
    def from_raw_fast(cls, raw_buffer: bytes | bytearray | memoryview) -> KernelRPCEnvelope:
        """
        Deserializes directly from a `bytes` or zero-copy `memoryview` buffer without intermediate
        string decoding or redundant dictionary copies.
        """
        if isinstance(raw_buffer, memoryview):
            data = json.loads(bytes(raw_buffer))
        else:
            data = json.loads(raw_buffer)
        return cls(
            trace_id=str(data["trace_id"]),
            agent_id=str(data["agent_id"]),
            domain=str(data["domain"]),
            epoch_ms=int(data["epoch_ms"]),
            layer0_policy_ok=int(data["layer0_policy_ok"]),
            layer1_runtime_ok=int(data["layer1_runtime_ok"]),
            payload_vector=tuple(float(x) for x in data["payload_vector"]),
            source_data_object=str(data.get("source_data_object", "urn:adam:kernel:telemetry:v30.1")),
            confidence_score=float(data.get("confidence_score", 0.95)),
            git_commit_hash=str(data.get("git_commit_hash", "v30.1-afos-kernel")),
            json_logic_version=str(data.get("json_logic_version", "1.0.0")),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        """Returns a canonical dictionary representation suitable for deterministic hashing."""
        return {
            "agent_id": self.agent_id,
            "confidence_score": self.confidence_score,
            "domain": self.domain,
            "epoch_ms": self.epoch_ms,
            "git_commit_hash": self.git_commit_hash,
            "json_logic_version": self.json_logic_version,
            "layer0_policy_ok": self.layer0_policy_ok,
            "layer1_runtime_ok": self.layer1_runtime_ok,
            "payload_vector": list(self.payload_vector),
            "source_data_object": self.source_data_object,
            "trace_id": self.trace_id,
        }

    def build_provenance_header(self, content_hash: str | None = None) -> ProvenanceHeader:
        """Constructs a validated PDIL `ProvenanceHeader` for W3C PROV-O audit compliance."""
        digest = content_hash or compute_deterministic_hash(self.to_canonical_dict())
        ts_iso = datetime.fromtimestamp(self.epoch_ms / 1000.0, tz=timezone.utc).isoformat()
        return ProvenanceHeader(
            git_commit_hash=self.git_commit_hash,
            timestamp=ts_iso,
            content_hash=digest,
            jsonLogic_version=self.json_logic_version,
            confidence_score=self.confidence_score,
            derivation_path=f"KernelRPC -> {self.domain} -> {self.agent_id}",
            source_data_object=self.source_data_object,
        )

    def check_grounding(self) -> bool:
        """Verifies non-empty `source_data_object` per W3C PROV-O requirements."""
        return bool(self.source_data_object and self.source_data_object.strip())


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Computes linear-interpolated percentile over a pre-sorted sequence of floats."""
    if not sorted_values:
        return 0.0
    if pct <= 0.0:
        return float(sorted_values[0])
    if pct >= 100.0:
        return float(sorted_values[-1])
    idx = (len(sorted_values) - 1) * (pct / 100.0)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return float(sorted_values[int(idx)])
    weight = idx - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def summarize_latencies(sorted_latencies_ms: Sequence[float], wall_time_ms: float) -> dict[str, float]:
    """Computes comprehensive SRE tail-latency, jitter, and throughput metrics."""
    if not sorted_latencies_ms:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
            "iqr": 0.0,
            "min": 0.0,
            "max": 0.0,
            "wall_time_ms": round(wall_time_ms, 3),
            "throughput_rps": 0.0,
        }
    p25 = percentile(sorted_latencies_ms, 25.0)
    p75 = percentile(sorted_latencies_ms, 75.0)
    stdev_val = statistics.stdev(sorted_latencies_ms) if len(sorted_latencies_ms) > 1 else 0.0
    throughput = (len(sorted_latencies_ms) / (wall_time_ms / 1000.0)) if wall_time_ms > 0 else 0.0
    return {
        "p50": round(percentile(sorted_latencies_ms, 50.0), 3),
        "p90": round(percentile(sorted_latencies_ms, 90.0), 3),
        "p95": round(percentile(sorted_latencies_ms, 95.0), 3),
        "p99": round(percentile(sorted_latencies_ms, 99.0), 3),
        "p99_9": round(percentile(sorted_latencies_ms, 99.9), 3),
        "mean": round(statistics.mean(sorted_latencies_ms), 3),
        "stdev": round(stdev_val, 3),
        "iqr": round(p75 - p25, 3),
        "min": round(float(sorted_latencies_ms[0]), 3),
        "max": round(float(sorted_latencies_ms[-1]), 3),
        "wall_time_ms": round(wall_time_ms, 3),
        "throughput_rps": round(throughput, 2),
    }


def verify_prov_o_ledger_chain(ledger_path: Path) -> tuple[bool, int, str]:
    """
    Verifies cryptographic integrity and monotonic C6 hash chaining of a hardened PROV-O JSONL ledger.
    Returns `(is_valid, verified_record_count, final_chain_hash_or_error)`.
    """
    if not ledger_path.exists():
        return True, 0, GENESIS_CHAIN_HASH

    prev_chain_hash = GENESIS_CHAIN_HASH
    count = 0
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("safety_gate") != 1:
                return False, count, f"Line {line_num}: Invalid safety_gate={record.get('safety_gate')}"
            if not record.get("source_data_object"):
                return False, count, f"Line {line_num}: Missing W3C PROV-O source_data_object"
            if record.get("prev_chain_hash") != prev_chain_hash:
                return (
                    False,
                    count,
                    f"Line {line_num}: Broken prev_chain_hash (expected {prev_chain_hash}, got {record.get('prev_chain_hash')})",
                )
            expected_chain_hash = hashlib.sha256(
                f"{prev_chain_hash}:{record['trace_id']}:{record['sha256']}:{record['epoch_ms']}".encode("utf-8")
            ).hexdigest()
            if record.get("chain_hash") != expected_chain_hash:
                return (
                    False,
                    count,
                    f"Line {line_num}: Chain hash mismatch (expected {expected_chain_hash}, got {record.get('chain_hash')})",
                )
            prev_chain_hash = expected_chain_hash
            count += 1

    return True, count, prev_chain_hash


class AsyncProvOBatchLedger:
    """
    Bounded non-blocking W3C PROV-O JSONL writer with:
      - Multiplicative Safety Gate (`Layer 0 x Layer 1`)
      - W3C PROV-O `source_data_object` Grounding Check
      - Monotonic C6 Hash Chaining (`prev_chain_hash -> chain_hash`)
      - Air-Gapped `QuarantineSink` diversion and 75/25 Barbell `StateManager` `LOCKDOWN` trigger on breach
      - Thread-offloaded batch disk flushing (`asyncio.to_thread`) to protect the event loop
    """

    def __init__(
        self,
        path: Path,
        max_queue: int = 2048,
        flush_batch_size: int = 32,
        quarantine_sink: QuarantineSink | None = None,
        state_manager: StateManager | None = None,
    ) -> None:
        self.path = path
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max_queue)
        self.flush_batch_size = flush_batch_size
        self.quarantine_sink = quarantine_sink
        self.state_manager = state_manager
        self._worker_task: asyncio.Task[None] | None = None
        self._chain_head: str = GENESIS_CHAIN_HASH
        self.committed_count: int = 0
        self.quarantined_count: int = 0

    @property
    def chain_head(self) -> str:
        return self._chain_head

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._worker_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._worker_task is not None:
            await self.queue.put(None)
            await self._worker_task
            self._worker_task = None

    async def _divert_to_quarantine(
        self, env: KernelRPCEnvelope, failure_category: str, reason: str
    ) -> None:
        self.quarantined_count += 1
        if self.state_manager is not None:
            self.state_manager.trigger_circuit_breaker(reason=f"{failure_category}: {reason}")
        if self.quarantine_sink is not None:
            await self.quarantine_sink.dump(
                failure_category=failure_category,
                source_node=env.agent_id,
                reason=reason,
                raw_payload=env.to_canonical_dict(),
                prov_o_trace={
                    "trace_id": env.trace_id,
                    "domain": env.domain,
                    "epoch_ms": env.epoch_ms,
                    "layer0_policy_ok": env.layer0_policy_ok,
                    "layer1_runtime_ok": env.layer1_runtime_ok,
                },
            )

    async def commit_event(self, env: KernelRPCEnvelope) -> str:
        # 1. Multiplicative safety gate: Layer 0 x Layer 1
        gate_product = env.safety_gate
        if gate_product == 0:
            reason = (
                f"Safety gate tripped for trace_id={env.trace_id} "
                f"(layer0={env.layer0_policy_ok}, layer1={env.layer1_runtime_ok})"
            )
            await self._divert_to_quarantine(env, "SAFETY_GATE_BREACH", reason)
            raise PermissionError(reason)

        # 2. W3C PROV-O grounding check
        if not env.check_grounding():
            reason = f"W3C PROV-O grounding violation: empty source_data_object for trace_id={env.trace_id}"
            await self._divert_to_quarantine(env, "PROV_O_GROUNDING_BREACH", reason)
            raise PermissionError(reason)

        # 3. Deterministic canonical SHA-256 content hash
        sha256_hex = compute_deterministic_hash(env.to_canonical_dict())
        header = env.build_provenance_header(content_hash=sha256_hex)

        event_payload = {
            "prov_type": "prov:Activity",
            "trace_id": env.trace_id,
            "agent_id": env.agent_id,
            "domain": env.domain,
            "epoch_ms": env.epoch_ms,
            "sha256": sha256_hex,
            "state_hash": header.hash_state(),
            "source_data_object": header.source_data_object,
            "derivation_path": header.derivation_path,
            "confidence_score": header.confidence_score,
            "safety_gate": gate_product,
        }
        await asyncio.wait_for(self.queue.put(event_payload), timeout=0.25)
        return sha256_hex

    @staticmethod
    def _write_lines_sync(path: Path, lines: list[str]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
            fh.flush()

    def _seal_record(self, item: dict[str, Any]) -> str:
        prev_hash = self._chain_head
        chain_hash = hashlib.sha256(
            f"{prev_hash}:{item['trace_id']}:{item['sha256']}:{item['epoch_ms']}".encode("utf-8")
        ).hexdigest()
        self._chain_head = chain_hash
        self.committed_count += 1
        sealed = {
            **item,
            "prev_chain_hash": prev_hash,
            "chain_hash": chain_hash,
        }
        return json.dumps(sealed, separators=(",", ":"), sort_keys=True)

    async def _flush_loop(self) -> None:
        batch: list[str] = []
        while True:
            item = await self.queue.get()
            if item is None:
                if batch:
                    await asyncio.to_thread(self._write_lines_sync, self.path, list(batch))
                    batch.clear()
                break

            batch.append(self._seal_record(item))

            # Drain immediately available items up to flush_batch_size without blocking
            while len(batch) < self.flush_batch_size:
                try:
                    next_item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if next_item is None:
                    await asyncio.to_thread(self._write_lines_sync, self.path, list(batch))
                    batch.clear()
                    return
                batch.append(self._seal_record(next_item))

            if len(batch) >= self.flush_batch_size or self.queue.empty():
                await asyncio.to_thread(self._write_lines_sync, self.path, list(batch))
                batch.clear()


@dataclass(frozen=True)
class HardenedWaveMetrics:
    """Detailed execution telemetry returned by `run_hardened_dag_wave_detailed`."""

    latencies_ms: list[float]
    exec_latencies_ms: list[float]
    queue_wait_ms: list[float]
    wall_time_ms: float
    committed_count: int
    quarantined_count: int
    chain_head: str
    chain_verified: bool
    system_status: str
    allocations: dict[str, float]



async def run_naive_dag_wave(raw_messages: list[bytes], ledger_path: Path) -> list[float]:
    """Simulates un-throttled fan-out with double JSON deserialization and synchronous disk I/O contention."""
    latencies_ms: list[float] = []

    async def _naive_worker(raw: bytes) -> None:
        t0 = time.perf_counter()
        intermediate = json.loads(raw.decode("utf-8"))
        re_encoded = json.dumps(intermediate)
        parsed = json.loads(re_encoded)
        digest = hashlib.sha256(re_encoded.encode("utf-8")).hexdigest()
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"trace_id": parsed["trace_id"], "sha256": digest}) + "\n")
            fh.flush()
        await asyncio.sleep(0.001)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    await asyncio.gather(*(_naive_worker(m) for m in raw_messages))
    return sorted(latencies_ms)


async def run_hardened_dag_wave_detailed(
    raw_messages: list[bytes],
    ledger_path: Path,
    max_concurrency: int = 32,
    quarantine_path: Path | None = None,
    allow_faults: bool = False,
) -> HardenedWaveMetrics:
    """
    Executes the hardened DAG scheduler wave with bounded concurrency, zero-copy envelope parsing,
    async batch PROV-O ledgering, monotonic C6 hash chaining, and QuarantineSink/StateManager hooks.
    """
    latencies_ms: list[float] = []
    exec_latencies_ms: list[float] = []
    queue_wait_ms: list[float] = []
    sem = asyncio.Semaphore(max_concurrency)
    q_path = quarantine_path or (ledger_path.parent / "quarantine.jsonl")
    quarantine_sink = QuarantineSink(sink_path=q_path)
    state_manager = StateManager(redis_url="redis://127.0.0.1:0/0")
    ledger = AsyncProvOBatchLedger(
        ledger_path,
        quarantine_sink=quarantine_sink,
        state_manager=state_manager,
    )
    await ledger.start()
    wave_t0 = time.perf_counter()

    async def _hardened_worker(raw: bytes) -> None:
        t0 = time.perf_counter()
        async with sem:
            t_exec = time.perf_counter()
            queue_wait_ms.append((t_exec - t0) * 1000.0)
            env = KernelRPCEnvelope.from_raw_fast(memoryview(raw))
            try:
                await ledger.commit_event(env)
            except PermissionError:
                if not allow_faults:
                    raise
            await asyncio.sleep(0.001)
            t_done = time.perf_counter()
            exec_latencies_ms.append((t_done - t_exec) * 1000.0)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    try:
        async with asyncio.TaskGroup() as tg:
            for m in raw_messages:
                tg.create_task(_hardened_worker(m))
    finally:
        await ledger.stop()

    wall_time_ms = (time.perf_counter() - wave_t0) * 1000.0
    chain_ok, _, _ = verify_prov_o_ledger_chain(ledger_path)

    return HardenedWaveMetrics(
        latencies_ms=sorted(latencies_ms),
        exec_latencies_ms=sorted(exec_latencies_ms),
        queue_wait_ms=sorted(queue_wait_ms),
        wall_time_ms=wall_time_ms,
        committed_count=ledger.committed_count,
        quarantined_count=ledger.quarantined_count,
        chain_head=ledger.chain_head,
        chain_verified=chain_ok,
        system_status=state_manager.system_status,
        allocations=dict(state_manager.allocations),
    )


async def run_hardened_dag_wave(
    raw_messages: list[bytes], ledger_path: Path, max_concurrency: int = 32
) -> list[float]:
    """
    Backward-compatible entry point returning sorted per-task latencies in milliseconds.
    """
    details = await run_hardened_dag_wave_detailed(
        raw_messages=raw_messages,
        ledger_path=ledger_path,
        max_concurrency=max_concurrency,
    )
    return details.latencies_ms


def build_synthetic_wave(
    agent_count: int = 80,
    adversarial_fault_count: int = 0,
) -> list[bytes]:
    """Builds a deterministic wave of serialized `KernelRPCEnvelope` payloads."""
    messages: list[bytes] = []
    for i in range(agent_count):
        is_fault = i < adversarial_fault_count
        layer0 = 0 if (is_fault and i % 2 == 0) else 1
        layer1 = 0 if (is_fault and i % 2 == 1) else 1
        messages.append(
            json.dumps(
                {
                    "trace_id": f"trc-20261106-{i:04d}",
                    "agent_id": f"agent-node-{i:03d}",
                    "domain": (
                        "enterprise" if i % 3 == 0 else "research" if i % 3 == 1 else "governance"
                    ),
                    "epoch_ms": 1794010000000 + i,
                    "layer0_policy_ok": layer0,
                    "layer1_runtime_ok": layer1,
                    "payload_vector": [round(float(x) * 0.015, 6) for x in range(32)],
                    "source_data_object": "urn:adam:kernel:telemetry:v30.1",
                    "confidence_score": 0.96,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    return messages


def evaluate_sre_invariants(
    naive_stats: dict[str, float],
    hardened_stats: dict[str, float],
    hardened_details: HardenedWaveMetrics,
    fault_wave_details: HardenedWaveMetrics,
    expected_agents: int,
    expected_faults: int,
) -> tuple[GateResult, IntegrationDecision]:
    """
    Evaluates deterministic SRE invariants via `core.evaluation.invariant_enforcer`
    and `core.evaluation.integration_verifier`.
    """
    invariant_checks = {
        "prov_o_chain_integrity": hardened_details.chain_verified
        and hardened_details.committed_count == expected_agents,
        "zero_nominal_quarantines": hardened_details.quarantined_count == 0
        and hardened_details.system_status == "NORMAL",
        "adversarial_fault_isolation": (
            fault_wave_details.quarantined_count == expected_faults
            and fault_wave_details.committed_count == (expected_agents - expected_faults)
            and fault_wave_details.chain_verified
        ),
        "barbell_lockdown_circuit_breaker": (
            fault_wave_details.system_status == "LOCKDOWN"
            and fault_wave_details.allocations.get("fortress") == 0.75
            and fault_wave_details.allocations.get("convex_hunt") == 0.00
        ),
        "bounded_p99_latency_slo": hardened_stats["p99"] < 250.0,
    }
    reasons = {
        "prov_o_chain_integrity": (
            f"chain_verified={hardened_details.chain_verified}, committed={hardened_details.committed_count}/{expected_agents}"
        ),
        "zero_nominal_quarantines": (
            f"quarantined={hardened_details.quarantined_count}, status={hardened_details.system_status}"
        ),
        "adversarial_fault_isolation": (
            f"quarantined={fault_wave_details.quarantined_count}/{expected_faults}, "
            f"committed={fault_wave_details.committed_count}/{expected_agents - expected_faults}"
        ),
        "barbell_lockdown_circuit_breaker": (
            f"status={fault_wave_details.system_status}, allocations={fault_wave_details.allocations}"
        ),
        "bounded_p99_latency_slo": f"hardened_p99={hardened_stats['p99']}ms (SLO < 250.0ms)",
    }

    gate_result = enforce_invariants(invariant_checks, reasons=reasons)
    decision = decide_integration(
        hard_gate_passed=gate_result.passed,
        scope_passed=True,
        policy_passed=invariant_checks["adversarial_fault_isolation"]
        and invariant_checks["barbell_lockdown_circuit_breaker"],
        provenance_passed=invariant_checks["prov_o_chain_integrity"],
        calibration_healthy=invariant_checks["bounded_p99_latency_slo"],
        confidence=0.96,
        threshold=0.85,
    )
    return gate_result, decision


async def main(
    agent_count: int = 80,
    max_concurrency: int = 32,
    adversarial_fault_count: int = 4,
    emit_stdout: bool = True,
) -> dict[str, Any]:
    raw_messages = build_synthetic_wave(agent_count=agent_count, adversarial_fault_count=0)
    fault_messages = build_synthetic_wave(
        agent_count=agent_count,
        adversarial_fault_count=min(adversarial_fault_count, agent_count),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        naive_ledger = tmp_path / "naive_prov.jsonl"
        hardened_ledger = tmp_path / "hardened_prov.jsonl"
        fault_ledger = tmp_path / "fault_prov.jsonl"
        fault_quarantine = tmp_path / "fault_quarantine.jsonl"

        naive_t0 = time.perf_counter()
        naive_lat = await run_naive_dag_wave(raw_messages, naive_ledger)
        naive_wall_ms = (time.perf_counter() - naive_t0) * 1000.0

        hardened_details = await run_hardened_dag_wave_detailed(
            raw_messages,
            hardened_ledger,
            max_concurrency=max_concurrency,
        )

        fault_details = await run_hardened_dag_wave_detailed(
            fault_messages,
            fault_ledger,
            max_concurrency=max_concurrency,
            quarantine_path=fault_quarantine,
            allow_faults=True,
        )

    naive_stats = summarize_latencies(naive_lat, naive_wall_ms)
    hardened_stats = summarize_latencies(hardened_details.latencies_ms, hardened_details.wall_time_ms)
    hardened_exec_stats = summarize_latencies(
        hardened_details.exec_latencies_ms, hardened_details.wall_time_ms
    )
    hardened_queue_stats = summarize_latencies(
        hardened_details.queue_wait_ms, hardened_details.wall_time_ms
    )

    gate_result, decision = evaluate_sre_invariants(
        naive_stats=naive_stats,
        hardened_stats=hardened_stats,
        hardened_details=hardened_details,
        fault_wave_details=fault_details,
        expected_agents=agent_count,
        expected_faults=min(adversarial_fault_count, agent_count),
    )

    report: dict[str, Any] = {
        "agent_count": agent_count,
        "max_concurrency": max_concurrency,
        "naive_ms": {
            "p50": naive_stats["p50"],
            "p90": naive_stats["p90"],
            "p95": naive_stats["p95"],
            "p99": naive_stats["p99"],
            "p99_9": naive_stats["p99_9"],
            "mean": naive_stats["mean"],
            "stdev": naive_stats["stdev"],
            "iqr": naive_stats["iqr"],
            "wall_time_ms": naive_stats["wall_time_ms"],
            "throughput_rps": naive_stats["throughput_rps"],
        },
        "hardened_ms": {
            "p50": hardened_stats["p50"],
            "p90": hardened_stats["p90"],
            "p95": hardened_stats["p95"],
            "p99": hardened_stats["p99"],
            "p99_9": hardened_stats["p99_9"],
            "mean": hardened_stats["mean"],
            "stdev": hardened_stats["stdev"],
            "iqr": hardened_stats["iqr"],
            "wall_time_ms": hardened_stats["wall_time_ms"],
            "throughput_rps": hardened_stats["throughput_rps"],
        },
        "hardened_active_exec_ms": {
            "p50": hardened_exec_stats["p50"],
            "p90": hardened_exec_stats["p90"],
            "p95": hardened_exec_stats["p95"],
            "p99": hardened_exec_stats["p99"],
            "p99_9": hardened_exec_stats["p99_9"],
            "mean": hardened_exec_stats["mean"],
            "stdev": hardened_exec_stats["stdev"],
            "iqr": hardened_exec_stats["iqr"],
        },
        "hardened_semaphore_queue_wait_ms": {
            "p50": hardened_queue_stats["p50"],
            "p90": hardened_queue_stats["p90"],
            "p95": hardened_queue_stats["p95"],
            "p99": hardened_queue_stats["p99"],
            "mean": hardened_queue_stats["mean"],
        },
        "prov_o_audit": {
            "committed_records": hardened_details.committed_count,
            "chain_verified": hardened_details.chain_verified,
            "final_chain_sha256": hardened_details.chain_head,
            "nominal_system_status": hardened_details.system_status,
            "nominal_allocations": hardened_details.allocations,
        },
        "adversarial_fault_drill": {
            "injected_faults": min(adversarial_fault_count, agent_count),
            "quarantined_records": fault_details.quarantined_count,
            "surviving_committed_records": fault_details.committed_count,
            "surviving_chain_verified": fault_details.chain_verified,
            "post_fault_system_status": fault_details.system_status,
            "post_fault_allocations": fault_details.allocations,
        },
        "governance_gate": {
            "invariants_passed": gate_result.passed,
            "integration_allowed": decision.allowed,
            "integration_reason": decision.reason,
            "invariants": [
                {"name": inv.name, "passed": inv.passed, "reason": inv.reason}
                for inv in gate_result.invariants
            ],
        },
    }
    if emit_stdout:
        print(json.dumps(report, indent=2))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic SRE Latency & Concurrency Benchmark Harness for adamvangrover/adam (/core_kernel)."
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=80,
        help="Number of concurrent agent envelopes in the DAG wave (default: 80).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Bounded semaphore concurrency limit for the hardened wave (default: 32).",
    )
    parser.add_argument(
        "--faults",
        type=int,
        default=4,
        help="Number of Layer0/Layer1 safety gate faults to inject in the adversarial drill (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to persist the JSON benchmark report.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args()
    result = asyncio.run(
        main(
            agent_count=cli_args.agents,
            max_concurrency=cli_args.concurrency,
            adversarial_fault_count=cli_args.faults,
        )
    )
    if cli_args.output is not None:
        cli_args.output.parent.mkdir(parents=True, exist_ok=True)
        cli_args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
