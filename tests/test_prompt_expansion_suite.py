"""
Test suite validating newly added prompt kits, orchestrators, guides,
maintenance kits, classifiers, labelers, evals, rubrics, and decision envelopes.
"""

import json
from pathlib import Path
import pytest
from jinja2 import Environment, meta

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_LIB = REPO_ROOT / "prompt_library"
MANIFEST_FILE = PROMPT_LIB / "prompt_manifest.json"
CATALOG_FILE = PROMPT_LIB / "MASTER_CATALOG.md"


def test_manifest_exists_and_valid():
    """Verify prompt_manifest.json exists, parses, and conforms to required structure."""
    assert MANIFEST_FILE.is_file(), f"Manifest missing at {MANIFEST_FILE}"
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("catalog_version") == "2.0.0"
    assert "prompts" in data
    prompts = data["prompts"]
    assert len(prompts) >= 15

    for p in prompts:
        assert "prompt_id" in p
        assert "name" in p
        assert "category" in p
        assert "filepath" in p
        assert "epistemic_level" in p
        target_path = REPO_ROOT / p["filepath"]
        assert target_path.is_file(), f"Referenced file does not exist: {p['filepath']}"
        assert target_path.stat().st_size > 100, f"Referenced file is too small/empty: {p['filepath']}"


def test_catalog_file_completeness():
    """Verify MASTER_CATALOG.md contains references to all prompt files in manifest."""
    assert CATALOG_FILE.is_file(), f"Catalog missing at {CATALOG_FILE}"
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        catalog_text = f.read()

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for p in data["prompts"]:
        assert p["prompt_id"] in catalog_text, f"Catalog missing prompt_id {p['prompt_id']}"
        rel_filename = Path(p["filepath"]).name
        assert rel_filename in catalog_text, f"Catalog missing file reference {rel_filename}"


@pytest.mark.parametrize(
    "filepath, required_sections",
    [
        (
            "prompt_library/spec_kits/SPEC_KIT_ENGINEERING_CORE.md",
            ["Role & System Identity", "Invariants", "Input Context Variables", "Spec-It Generation Protocol"],
        ),
        (
            "prompt_library/spec_kits/SPEC_KIT_FINANCIAL_CREDIT.md",
            ["Role & Domain Mandate", "Quantitative & Regulatory Invariants", "Spec-It Protocol for Credit"],
        ),
        (
            "prompt_library/spec_kits/SPEC_KIT_SWARM_AGENT.md",
            ["Role & System Identity", "Swarm Invariants", "Spec-It Protocol for Swarms"],
        ),
        (
            "prompt_library/AOPL-v2.0/orchestration/SWARM_ROLLUP_HIERARCHICAL_AGGREGATOR.md",
            ["Role & Identity", "Invariants & Multiplicative Safety", "Step-by-Step Execution Protocol"],
        ),
        (
            "prompt_library/AOPL-v2.0/orchestration/PRODUCTION_DEPLOYMENT_ORCHESTRATOR.md",
            ["Role & Identity", "Invariants & Safety Primacy", "Step-by-Step Deployment Protocol"],
        ),
        (
            "prompt_library/AOPL-v2.0/orchestration/MULTI_AGENT_CONSENSUS_DISPATCHER.md",
            ["Role & Identity", "Invariants & Dispatch Directives", "Execution & Dispatch Protocol"],
        ),
        (
            "prompt_library/guides/BUILD_GUIDE_AUTONOMOUS_PIPELINE.md",
            ["Scope & Objective", "Monorepo Structural Rules", "Step-by-Step Architecture Scaffolding Protocol"],
        ),
        (
            "prompt_library/guides/FIELD_GUIDE_INCIDENT_TRIAGE.md",
            ["Scope & Objective", "Emergency Escalation Matrix", "Incident Triage Procedures"],
        ),
        (
            "prompt_library/run_kits/RUN_KIT_DAILY_OPERATIONS.md",
            ["Role & Identity", "Invariants & Execution Mandates", "Daily Runbook Sequence"],
        ),
        (
            "prompt_library/run_kits/MAINTENANCE_KIT_REFACTORING_HEALING.md",
            ["Role & Identity", "Invariants & Guardrails", "Maintenance Execution Protocol"],
        ),
        (
            "prompt_library/classifiers_labelers/CLASSIFIER_EPISTEMIC_GROUNDING.md",
            ["Role & Task Definition", "Epistemic Taxonomy Definitions", "Epistemic Inheritance Law"],
        ),
        (
            "prompt_library/classifiers_labelers/LABELER_COVENANT_EXTRACTOR.md",
            ["Role & Task Definition", "Invariants & Extraction Rules", "Extraction Protocol"],
        ),
        (
            "prompt_library/classifiers_labelers/CLASSIFIER_ANOMALY_DRIFT_DETECTOR.md",
            ["Role & Task Definition", "Invariants & Calibration Directives", "Detection Protocol"],
        ),
        (
            "prompt_library/evals_rubrics/BENCHMARK_FIDUCIARY_FITNESS.md",
            ["Role & Benchmark Identity", "Benchmark Scenarios", "Evaluation Protocol & Scoring Rubric"],
        ),
        (
            "prompt_library/evals_rubrics/EVAL_LLM_AS_JUDGE_ARBITER.md",
            ["Role & Task Definition", "Invariants & Multi-Criteria Rubric", "Position Bias Elimination"],
        ),
        (
            "prompt_library/evals_rubrics/SCORING_RUBRIC_CODE_SAFETY.md",
            ["Role & Objective", "Multiplicative Safety Gate Framework", "Layer 0 Hard Invariants"],
        ),
        (
            "prompt_library/decision_envelopes/DECISION_ENVELOPE_CAPITAL_ALLOCATION.md",
            ["Role & Identity", "Invariants & Safety Primacy", "Control Barrier Function"],
        ),
        (
            "prompt_library/decision_envelopes/DECISION_ENVELOPE_MODEL_RISK_GOVERNANCE.md",
            ["Role & Identity", "Governance Directives & Invariants", "Automated Kill-Switch Triggers"],
        ),
    ],
)
def test_prompt_file_structure_and_sections(filepath, required_sections):
    """Verify that each prompt file exists and contains expected architectural sections."""
    target_path = REPO_ROOT / filepath
    assert target_path.is_file(), f"Missing file: {filepath}"
    content = target_path.read_text(encoding="utf-8")
    for sec in required_sections:
        assert sec in content, f"File {filepath} missing expected section: '{sec}'"


def test_jinja2_variable_integrity():
    """Verify that all Jinja2 variables declared in the prompt files can be parsed cleanly."""
    env = Environment(autoescape=True)
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for p in manifest["prompts"]:
        target_path = REPO_ROOT / p["filepath"]
        content = target_path.read_text(encoding="utf-8")
        parsed_content = env.parse(content)
        variables = meta.find_undeclared_variables(parsed_content)

        # Ensure declared context_variables are found or valid
        for var in p.get("context_variables", []):
            assert var in variables, f"Variable '{var}' listed in manifest for {p['filepath']} but not found in Jinja AST"
