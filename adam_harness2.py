"""
ADAM OS Harness² Architecture (arXiv:2610.14224) - Adapted for 2nd-Line Credit Risk Oversight.

This module implements the 8-component file-mounted harness, Level 1 Task-Level Recursion
(Propose-Probe-Compose), Level 2 Harnessing-Level Recursion (EXPERIENCE.md library),
and SR 11-7 model risk management governance locks with W3C PROV-O telemetry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from core.provenance.prov import ProvActivity, ProvAgent, ProvEntity

logger = logging.getLogger(__name__)

MUTABLE_COMPONENTS = (
    "system_prompt",
    "memory",
    "skills",
    "tool_guidance",
    "scripts",
    "subagents",
    "plugins",
)
LOCKED_COMPONENTS = ("guardrails",)  # SR 11-7 invariant: Editor/Composer cannot alter guardrails


class Harness2TelemetryLogger:
    """Structured Telemetry Logger for Harness² SR 11-7 Auditability."""

    def __init__(
        self,
        sink: Callable[[str, dict], None] | None = None,
        log_file: Path | None = None,
    ) -> None:
        self.sink = sink
        self.log_file = log_file

    def emit(self, event_type: str, payload: dict) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        logger.debug(f"[Telemetry] {event_type}: {payload}")
        if self.sink:
            try:
                self.sink(event_type, payload)
            except Exception as err:
                logger.warning(f"Telemetry sink exception: {err}")
        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as err:
                logger.warning(f"Telemetry log write exception: {err}")


def create_w3c_prov_bundle(
    task_id: str,
    domain: str,
    activity_name: str,
    attributes: dict,
) -> tuple[dict, str]:
    """Generates a W3C PROV-O compliant provenance record and SHA-256 digest."""
    now_iso = datetime.now(UTC).isoformat()
    agent = ProvAgent(
        id="agent:harness2_controller",
        agent_type="software",
        attributes={"framework": "ADAM_OS_v30.1", "compliance": "SR_11_7"},
    )
    activity = ProvActivity(
        id=f"activity:{activity_name}_{task_id}_{int(datetime.now(UTC).timestamp())}",
        startTime=now_iso,
        endTime=now_iso,
        attributes=attributes,
    )
    entity = ProvEntity(
        id=f"entity:harness_state_{task_id}",
        attributes={"domain": domain, "task_id": task_id},
    )

    bundle = {
        "prov:agent": agent.model_dump(),
        "prov:activity": activity.model_dump(),
        "prov:entity": entity.model_dump(),
    }
    bundle_bytes = json.dumps(bundle, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(bundle_bytes).hexdigest()
    bundle["sha256_digest"] = digest
    return bundle, digest


@dataclass
class HarnessConfig:
    """Represents the 8 file-mounted components around a frozen model."""

    system_prompt: str
    guardrails: str  # Locked against self-modification
    memory: str = ""
    tool_guidance: str = ""
    subagents: str = ""
    skills: str = ""
    scripts: dict[str, str] = field(default_factory=dict)  # filename -> python code
    plugins: list[str] = field(default_factory=list)  # middleware validator names

    def apply_patch(self, patch: dict) -> HarnessConfig:
        """Applies candidate edits while enforcing immutable guardrails and append-only plugins."""
        updated = copy.deepcopy(self)
        for key, value in patch.items():
            if key in LOCKED_COMPONENTS:
                raise PermissionError(
                    f"Governance violation: Component '{key}' is immutable under SR 11-7 policy."
                )
            if key == "plugins" and isinstance(value, list):
                # Append-only enforcement for middleware validation plugins
                existing = set(updated.plugins)
                updated.plugins.extend([p for p in value if p not in existing])
            elif key in MUTABLE_COMPONENTS:
                setattr(updated, key, value)
        return updated

    def fingerprint(self) -> str:
        """Computes a SHA-256 digest of the harness configuration."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def mount_to_disk(self, directory: Path) -> None:
        """Mounts the 8 components as files for executor inspection."""
        directory.mkdir(parents=True, exist_ok=True)
        for comp in (
            "system_prompt",
            "guardrails",
            "memory",
            "tool_guidance",
            "subagents",
            "skills",
        ):
            (directory / f"{comp}.md").write_text(getattr(self, comp), encoding="utf-8")
        scripts_dir = directory / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        for fname, code in self.scripts.items():
            (scripts_dir / fname).write_text(code, encoding="utf-8")
        (directory / "plugins.json").write_text(
            json.dumps(self.plugins, indent=2), encoding="utf-8"
        )


@dataclass
class ExecutionTrace:
    """Captures runtime trajectory observables and performance metrics."""

    harness_id: str
    candidate_type: Literal["base", "invariant", "domain", "composed"]
    output: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_errors: int = 0
    verification_passed: bool = True
    latency_ms: float = 0.0
    provenance_digest: str = ""


@dataclass
class ExperienceRecord:
    """Record representing an adopted harness patch and lessons learned."""

    task_id: str
    domain: str
    adopted_patch: dict
    lessons_markdown: str  # Written to EXPERIENCE.md


class ExperienceLibrary:
    """Level-2 Harnessing Recursion: Persistent library of adopted harnesses & lessons."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def retrieve_lessons(self, domain: str, limit: int = 5) -> str:
        """Retrieves past domain lessons to condition Propose and Compose steps."""
        lessons = []
        matching_files = sorted(self.root_dir.glob(f"{domain}_*/EXPERIENCE.md"))
        for exp_file in matching_files[-limit:]:
            lessons.append(exp_file.read_text(encoding="utf-8"))
        return (
            "\n\n---\n\n".join(lessons)
            if lessons
            else "No prior domain lessons recorded."
        )

    def commit(self, record: ExperienceRecord, adopted_harness: HarnessConfig) -> Path:
        """Persists the adopted harness configuration, EXPERIENCE.md, patch.json, and provenance."""
        entry_dir = self.root_dir / f"{record.domain}_{record.task_id}"
        adopted_harness.mount_to_disk(entry_dir / "adopted_harness")
        (entry_dir / "EXPERIENCE.md").write_text(
            record.lessons_markdown, encoding="utf-8"
        )
        (entry_dir / "patch.json").write_text(
            json.dumps(record.adopted_patch, indent=2), encoding="utf-8"
        )

        prov_bundle, _digest = create_w3c_prov_bundle(
            task_id=record.task_id,
            domain=record.domain,
            activity_name="commit_experience",
            attributes={
                "adopted_fingerprint": adopted_harness.fingerprint(),
                "patch_keys": list(record.adopted_patch.keys()),
            },
        )
        (entry_dir / "provenance_bundle.json").write_text(
            json.dumps(prov_bundle, indent=2), encoding="utf-8"
        )
        return entry_dir


class Harness2Controller:
    """
    Task-Level Recursion (Propose - Probe - Compose) and Harnessing-Level Recursion Controller.
    """

    def __init__(
        self,
        base_harness: HarnessConfig,
        library: ExperienceLibrary,
        executor_fn: Callable[[HarnessConfig, str], ExecutionTrace],
        editor_fn: Callable[
            [str, ExecutionTrace, str, int], tuple[list[dict], list[dict]]
        ],
        composer_fn: Callable[
            [str, ExecutionTrace, list[ExecutionTrace], str], dict
        ],
        reflector_fn: Callable[[str, ExecutionTrace, ExecutionTrace, dict], str],
        telemetry_logger: Harness2TelemetryLogger | None = None,
    ) -> None:
        self.base_harness = base_harness
        self.library = library
        self.executor_fn = executor_fn
        self.editor_fn = editor_fn
        self.composer_fn = composer_fn
        self.reflector_fn = reflector_fn
        self.telemetry = telemetry_logger or Harness2TelemetryLogger()

    def run_task(
        self,
        task_id: str,
        domain: str,
        task_prompt: str,
        k: int = 1,
        mode: Literal["parallel", "sequential"] = "parallel",
    ) -> ExecutionTrace:
        """
        Executes Task-Level Recursion (Propose-Probe-Compose) and updates Level-2
        Harnessing Recursion (EXPERIENCE.md library). Always starts from clean baseline H_0.
        """
        # Invariant: Every task resets to the clean canonical Base Harness H_0
        current_harness = copy.deepcopy(self.base_harness)
        prior_lessons = self.library.retrieve_lessons(domain=domain)

        # Step 0: Reference execution on H_0
        ref_trace = self.executor_fn(current_harness, task_prompt)
        ref_trace.candidate_type = "base"
        _, ref_digest = create_w3c_prov_bundle(
            task_id=task_id,
            domain=domain,
            activity_name="reference_execution",
            attributes={"harness_id": current_harness.fingerprint()},
        )
        ref_trace.provenance_digest = ref_digest

        self.telemetry.emit(
            "harness2.reference_executed",
            {
                "task_id": task_id,
                "domain": domain,
                "base_fingerprint": current_harness.fingerprint(),
                "prov_digest": ref_digest,
            },
        )

        accumulated_patch: dict = {}

        if mode == "parallel":
            # Propose k invariant + k domain patches simultaneously
            inv_patches, dom_patches = self.editor_fn(
                task_prompt, ref_trace, prior_lessons, k
            )
            probe_traces: list[ExecutionTrace] = []

            for patch in inv_patches:
                cand_h = current_harness.apply_patch(patch)
                trace = self.executor_fn(cand_h, task_prompt)
                trace.candidate_type = "invariant"
                _, p_digest = create_w3c_prov_bundle(
                    task_id=task_id,
                    domain=domain,
                    activity_name="probe_invariant",
                    attributes={"harness_id": cand_h.fingerprint()},
                )
                trace.provenance_digest = p_digest
                probe_traces.append(trace)

            for patch in dom_patches:
                cand_h = current_harness.apply_patch(patch)
                trace = self.executor_fn(cand_h, task_prompt)
                trace.candidate_type = "domain"
                _, p_digest = create_w3c_prov_bundle(
                    task_id=task_id,
                    domain=domain,
                    activity_name="probe_domain",
                    attributes={"harness_id": cand_h.fingerprint()},
                )
                trace.provenance_digest = p_digest
                probe_traces.append(trace)

            self.telemetry.emit(
                "harness2.probes_evaluated",
                {
                    "task_id": task_id,
                    "probe_count": len(probe_traces),
                    "mode": mode,
                },
            )

            # Compose final harness from behavioral contrasts across all 2k probes
            accumulated_patch = self.composer_fn(
                task_prompt, ref_trace, probe_traces, prior_lessons
            )
            final_harness = current_harness.apply_patch(accumulated_patch)
            final_trace = self.executor_fn(final_harness, task_prompt)
            final_trace.candidate_type = "composed"

        elif mode == "sequential":
            # Carry refined harness through k successive Propose-Probe-Compose steps
            active_trace = ref_trace
            for _step_i in range(k):
                inv_patches, dom_patches = self.editor_fn(
                    task_prompt, active_trace, prior_lessons, 1
                )
                inv_trace = self.executor_fn(
                    current_harness.apply_patch(inv_patches[0]), task_prompt
                )
                inv_trace.candidate_type = "invariant"

                dom_trace = self.executor_fn(
                    current_harness.apply_patch(dom_patches[0]), task_prompt
                )
                dom_trace.candidate_type = "domain"

                step_patch = self.composer_fn(
                    task_prompt, active_trace, [inv_trace, dom_trace], prior_lessons
                )
                current_harness = current_harness.apply_patch(step_patch)
                accumulated_patch.update(step_patch)
                active_trace = self.executor_fn(current_harness, task_prompt)
                active_trace.candidate_type = "composed"

            final_harness = current_harness
            final_trace = active_trace

        _, final_digest = create_w3c_prov_bundle(
            task_id=task_id,
            domain=domain,
            activity_name="composed_execution",
            attributes={
                "harness_id": final_harness.fingerprint(),
                "mode": mode,
                "k": k,
            },
        )
        final_trace.provenance_digest = final_digest

        self.telemetry.emit(
            "harness2.composed_adopted",
            {
                "task_id": task_id,
                "final_fingerprint": final_harness.fingerprint(),
                "patch_keys": list(accumulated_patch.keys()),
                "prov_digest": final_digest,
            },
        )

        # Level 2: Experience Reflection -> Commit adopted harness + EXPERIENCE.md
        lessons_md = self.reflector_fn(
            task_prompt, ref_trace, final_trace, accumulated_patch
        )
        self.library.commit(
            ExperienceRecord(
                task_id=task_id,
                domain=domain,
                adopted_patch=accumulated_patch,
                lessons_markdown=lessons_md,
            ),
            adopted_harness=final_harness,
        )

        self.telemetry.emit(
            "harness2.experience_committed",
            {
                "task_id": task_id,
                "domain": domain,
            },
        )
        return final_trace
