"""
Unit tests for ADAM OS Harness² Architecture (adam_harness2.py),
Stage Prompts, Credit Benchmark Adapter, and Telemetry Logger.
"""

import json
from pathlib import Path
import pytest

from adam_harness2 import (
    ExecutionTrace,
    ExperienceLibrary,
    ExperienceRecord,
    Harness2Controller,
    Harness2TelemetryLogger,
    HarnessConfig,
)
from benchmarks.credit_bench import CreditBenchAdapter
from prompts.harness2_stage_prompts import (
    COMPOSER_STAGE_PROMPT,
    EDITOR_STAGE_PROMPT,
    REFLECTOR_STAGE_PROMPT,
    format_composer_prompt,
    format_editor_prompt,
    format_reflector_prompt,
    parse_composer_response,
    parse_editor_response,
)
from scripts.run_harness2_credit_bench import run_benchmark


@pytest.fixture
def sample_base_harness() -> HarnessConfig:
    return HarnessConfig(
        system_prompt="You are a 2nd-line Credit Risk Officer.",
        guardrails="LOCKED: SR 11-7 Risk Policy limits net leverage to 6.5x.",
        memory="Initial deal memory.",
        tool_guidance="Use EDGAR and financial calculator scripts.",
        subagents="Covenant subagent",
        skills="BSL credit analysis",
        scripts={"leverage_calc.py": "def calc_leverage(debt, ebitda): return debt / ebitda"},
        plugins=["schema_validator"],
    )


def test_harness_config_patching_and_fingerprint(sample_base_harness: HarnessConfig):
    patch = {
        "memory": "Updated deal memory with EBITDA add-backs.",
        "skills": "BSL credit analysis + LME triage",
    }
    patched = sample_base_harness.apply_patch(patch)

    assert patched.memory == "Updated deal memory with EBITDA add-backs."
    assert patched.skills == "BSL credit analysis + LME triage"
    assert patched.guardrails == sample_base_harness.guardrails
    assert len(patched.fingerprint()) == 12


def test_guardrail_lock_permission_error(sample_base_harness: HarnessConfig):
    illegal_patch = {
        "guardrails": "WEAKENED: Allow net leverage up to 10x without approval."
    }
    with pytest.raises(PermissionError) as exc_info:
        sample_base_harness.apply_patch(illegal_patch)

    assert "Component 'guardrails' is immutable" in str(exc_info.value)


def test_append_only_plugins(sample_base_harness: HarnessConfig):
    plugin_patch = {"plugins": ["accounting_identity_checker", "schema_validator"]}
    patched = sample_base_harness.apply_patch(plugin_patch)

    assert patched.plugins == ["schema_validator", "accounting_identity_checker"]


def test_harness_mount_to_disk(sample_base_harness: HarnessConfig, tmp_path: Path):
    target_dir = tmp_path / "mounted_harness"
    sample_base_harness.mount_to_disk(target_dir)

    assert (target_dir / "system_prompt.md").read_text(encoding="utf-8") == sample_base_harness.system_prompt
    assert (target_dir / "guardrails.md").read_text(encoding="utf-8") == sample_base_harness.guardrails
    assert (target_dir / "scripts" / "leverage_calc.py").exists()

    plugins = json.loads((target_dir / "plugins.json").read_text(encoding="utf-8"))
    assert plugins == ["schema_validator"]


def test_experience_library_commit_and_retrieval(sample_base_harness: HarnessConfig, tmp_path: Path):
    lib_dir = tmp_path / "exp_lib"
    library = ExperienceLibrary(lib_dir)

    record = ExperienceRecord(
        task_id="tmt_deal_1",
        domain="credit_tmt",
        adopted_patch={"skills": "TMT EBITDA add-back playbook"},
        lessons_markdown="# Lessons\nAlways inspect capitalized software R&D add-backs.",
    )

    committed_path = library.commit(record, sample_base_harness)
    assert committed_path.exists()
    assert (committed_path / "EXPERIENCE.md").exists()
    assert (committed_path / "provenance_bundle.json").exists()

    lessons = library.retrieve_lessons(domain="credit_tmt")
    assert "Always inspect capitalized software R&D add-backs" in lessons


def test_telemetry_logger_and_controller_integration(sample_base_harness: HarnessConfig, tmp_path: Path):
    emitted_events = []

    def custom_sink(event_type: str, payload: dict):
        emitted_events.append((event_type, payload))

    log_file = tmp_path / "telemetry.jsonl"
    telemetry = Harness2TelemetryLogger(sink=custom_sink, log_file=log_file)
    library = ExperienceLibrary(tmp_path / "exp_lib")

    def mock_executor(harness: HarnessConfig, prompt: str) -> ExecutionTrace:
        return ExecutionTrace(
            harness_id=harness.fingerprint(),
            candidate_type="base",
            output="Credit analysis with 10-K Section 7 covenant exhibits.",
            verification_passed=True,
        )

    def mock_editor(prompt: str, trace: ExecutionTrace, lessons: str, k: int):
        return [{"memory": "inv_patch"}], [{"skills": "dom_patch"}]

    def mock_composer(prompt: str, ref: ExecutionTrace, probes: list[ExecutionTrace], lessons: str):
        return {"skills": "dom_patch"}

    def mock_reflector(prompt: str, ref: ExecutionTrace, final: ExecutionTrace, patch: dict):
        return "# Reflection\nAdopted domain patch."

    controller = Harness2Controller(
        base_harness=sample_base_harness,
        library=library,
        executor_fn=mock_executor,
        editor_fn=mock_editor,
        composer_fn=mock_composer,
        reflector_fn=mock_reflector,
        telemetry_logger=telemetry,
    )

    trace = controller.run_task(
        task_id="task_telem_01",
        domain="credit_bsl",
        task_prompt="Evaluate leverage for Acme Corp.",
        k=1,
        mode="parallel",
    )

    assert trace.candidate_type == "composed"
    assert len(emitted_events) >= 4
    event_types = [e[0] for e in emitted_events]
    assert "harness2.reference_executed" in event_types
    assert "harness2.probes_evaluated" in event_types
    assert "harness2.composed_adopted" in event_types
    assert "harness2.experience_committed" in event_types

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 4


def test_stage_prompt_formatting_and_parsing():
    editor_prompt = format_editor_prompt(
        task_prompt="Evaluate Acme Corp leverage",
        ref_trace="Trace object str",
        prior_lessons="Prior lesson text",
        k=2,
    )
    assert "Evaluate Acme Corp leverage" in editor_prompt
    assert "Candidate Pairs Count (k): 2" in editor_prompt

    composer_prompt = format_composer_prompt(
        task_prompt="Evaluate Acme Corp leverage",
        ref_trace="Trace ref",
        probe_traces=["Probe 1", "Probe 2"],
        prior_lessons="Prior lesson text",
    )
    assert "NO-GROUND-TRUTH BEHAVIORAL SIGNALS TO EVALUATE" in composer_prompt
    assert "Probe 1" in composer_prompt

    reflector_prompt = format_reflector_prompt(
        domain="credit_tmt",
        task_id="task_001",
        task_prompt="Evaluate Acme Corp leverage",
        ref_trace="Trace ref",
        final_trace="Trace final",
        adopted_patch={"skills": "TMT playbook"},
    )
    assert "Experience Reflection: credit_tmt / Task task_001" in reflector_prompt

    inv, dom = parse_editor_response(
        '{"invariant_patches": [{"tool_guidance": "strict"}], "domain_patches": [{"skills": "tmt"}]}'
    )
    assert len(inv) == 1 and inv[0]["tool_guidance"] == "strict"
    assert len(dom) == 1 and dom[0]["skills"] == "tmt"

    inv_err, dom_err = parse_editor_response("invalid json text")
    assert inv_err == [] and dom_err == []

    comp_patch = parse_composer_response('{"skills": "composed_skill"}')
    assert comp_patch.get("skills") == "composed_skill"

    comp_err = parse_composer_response("invalid json text")
    assert comp_err == {}


def test_run_benchmark_execution(tmp_path: Path):
    output_dir = tmp_path / "bench_run"
    summary = run_benchmark(
        output_dir=str(output_dir),
        k=1,
        mode="parallel",
        domain="credit_bsl",
    )

    assert summary["dataset_size"] == 3
    assert summary["k"] == 1
    assert summary["mode"] == "parallel"
    assert len(summary["results"]) == 3
    assert (output_dir / "benchmark_summary.json").exists()


def test_credit_bench_adapter(sample_base_harness: HarnessConfig):
    adapter = CreditBenchAdapter()
    trace = ExecutionTrace(
        harness_id=sample_base_harness.fingerprint(),
        candidate_type="composed",
        output="Inspected 10-K exhibit 99.1 and Section 7 covenant table for EBITDA calculation.",
        tool_calls=[{"name": "leverage_calc.py", "status": "success"}],
        tool_errors=0,
        verification_passed=True,
    )

    scores = adapter.evaluate_trace(trace, sample_base_harness)

    assert scores.tool_script_determinism == 1.0
    assert scores.source_grounding_coverage > 0.5
    assert scores.middleware_plugin_pass_rate == 1.0
    assert scores.overall_score > 0.8
