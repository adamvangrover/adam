"""
Unit tests for ADAM OS Harness² Architecture (adam_harness2.py),
Stage Prompts, and Credit Benchmark Adapter.
"""

import json
from pathlib import Path
import pytest

from adam_harness2 import (
    ExecutionTrace,
    ExperienceLibrary,
    ExperienceRecord,
    Harness2Controller,
    HarnessConfig,
)
from benchmarks.credit_bench import CreditBenchAdapter
from prompts.harness2_stage_prompts import (
    COMPOSER_STAGE_PROMPT,
    EDITOR_STAGE_PROMPT,
    REFLECTOR_STAGE_PROMPT,
)


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


def test_harness2_controller_parallel_mode(sample_base_harness: HarnessConfig, tmp_path: Path):
    library = ExperienceLibrary(tmp_path / "exp_lib")

    def mock_executor(harness: HarnessConfig, prompt: str) -> ExecutionTrace:
        return ExecutionTrace(
            harness_id=harness.fingerprint(),
            candidate_type="base",
            output="Executed credit risk assessment with 10-K exhibit review and covenant verification.",
            tool_calls=[{"name": "leverage_calc", "args": {"debt": 100, "ebitda": 20}}],
            tool_errors=0,
            verification_passed=True,
        )

    def mock_editor(prompt: str, trace: ExecutionTrace, lessons: str, k: int):
        inv = [{"tool_guidance": "Strict verification mode"}]
        dom = [{"skills": "Enhanced SEC 10-K footnote triage"}]
        return inv, dom

    def mock_composer(prompt: str, ref: ExecutionTrace, probes: list[ExecutionTrace], lessons: str):
        return {"skills": "Enhanced SEC 10-K footnote triage"}

    def mock_reflector(prompt: str, ref: ExecutionTrace, final: ExecutionTrace, patch: dict):
        return "# Reflection\nAdopted 10-K footnote triage skill."

    controller = Harness2Controller(
        base_harness=sample_base_harness,
        library=library,
        executor_fn=mock_executor,
        editor_fn=mock_editor,
        composer_fn=mock_composer,
        reflector_fn=mock_reflector,
    )

    trace = controller.run_task(
        task_id="task_001",
        domain="credit_bsl",
        task_prompt="Evaluate leverage for Acme Corp.",
        k=1,
        mode="parallel",
    )

    assert trace.candidate_type == "composed"
    assert trace.provenance_digest != ""
    recorded_lessons = library.retrieve_lessons("credit_bsl")
    assert "Adopted 10-K footnote triage skill" in recorded_lessons


def test_harness2_controller_sequential_mode(sample_base_harness: HarnessConfig, tmp_path: Path):
    library = ExperienceLibrary(tmp_path / "exp_lib")

    def mock_executor(harness: HarnessConfig, prompt: str) -> ExecutionTrace:
        return ExecutionTrace(
            harness_id=harness.fingerprint(),
            candidate_type="base",
            output="Credit analysis output for 10-K Section 7 covenants.",
            tool_calls=[{"name": "covenant_checker"}],
            tool_errors=0,
            verification_passed=True,
        )

    def mock_editor(prompt: str, trace: ExecutionTrace, lessons: str, k: int):
        return [{"memory": "Step edit invariant"}], [{"memory": "Step edit domain"}]

    def mock_composer(prompt: str, ref: ExecutionTrace, probes: list[ExecutionTrace], lessons: str):
        return {"memory": "Composed step memory"}

    def mock_reflector(prompt: str, ref: ExecutionTrace, final: ExecutionTrace, patch: dict):
        return "# Sequential Reflection\nCompleted 2 steps."

    controller = Harness2Controller(
        base_harness=sample_base_harness,
        library=library,
        executor_fn=mock_executor,
        editor_fn=mock_editor,
        composer_fn=mock_composer,
        reflector_fn=mock_reflector,
    )

    trace = controller.run_task(
        task_id="task_002",
        domain="credit_lbo",
        task_prompt="Evaluate LBO covenant breach risk.",
        k=2,
        mode="sequential",
    )

    assert trace.candidate_type == "composed"
    assert trace.provenance_digest != ""


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


def test_stage_prompts_content():
    assert "guardrails" in EDITOR_STAGE_PROMPT
    assert "SR 11-7" in EDITOR_STAGE_PROMPT
    assert "Tool & Script Determinism" in COMPOSER_STAGE_PROMPT
    assert "EXPERIENCE.md" in REFLECTOR_STAGE_PROMPT
