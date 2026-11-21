"""Unit and integration verification tests for the Multi-Agent Engineering Architecture

and Operational Blueprint Reference Manual & Prompt Suite.
"""

from pathlib import Path
import re
import pytest

MANUAL_PATH = Path("docs/reference/multi_agent_engineering_manual.md")
PROMPT_SUITE_PATH = Path("prompt_library/AOPL-v2.0/professional_outcomes/multi_agent_engineering_spec_it_and_decision_envelope_prompt_suite.md")
REFERENCE_README_PATH = Path("docs/reference/README.md")


def test_files_exist():
    """Verify that both manual and prompt suite files exist on disk."""
    assert MANUAL_PATH.exists(), f"Manual file missing at {MANUAL_PATH}"
    assert PROMPT_SUITE_PATH.exists(), f"Prompt suite file missing at {PROMPT_SUITE_PATH}"
    assert REFERENCE_README_PATH.exists(), f"Reference index missing at {REFERENCE_README_PATH}"


def test_reference_readme_indexing():
    """Ensure multi_agent_engineering_manual.md is indexed in docs/reference/README.md."""
    content = REFERENCE_README_PATH.read_text(encoding="utf-8")
    assert "multi_agent_engineering_manual.md" in content
    assert "Multi-Agent Engineering Architecture" in content


@pytest.mark.parametrize("filepath", [MANUAL_PATH, PROMPT_SUITE_PATH])
def test_domains_presence(filepath: Path):
    """Verify all 5 lifecycle domains exist in both documents."""
    content = filepath.read_text(encoding="utf-8")
    expected_domains = [
        "Domain 1: Spec-IT Kits",
        "Domain 2: Production & Swarm Roll-Ups",
        "Domain 3: Build Guides, Field Guides, Maintenance & Run Kits",
        "Domain 4: Internal Classifiers & Data Labelers",
        "Domain 5: Evals, Benchmarks, Scoring Rubrics & Decision Envelopes",
    ]
    for domain in expected_domains:
        assert domain in content, f"Missing domain header: {domain} in {filepath}"


@pytest.mark.parametrize("filepath", [MANUAL_PATH, PROMPT_SUITE_PATH])
def test_prompt_specifications_presence(filepath: Path):
    """Verify all prompt specifications and role definitions are present."""
    content = filepath.read_text(encoding="utf-8")
    expected_prompts = [
        "SPEC-IT-CORE",
        "SPEC-IT-INTERFACE",
        "SWARM-ROLLUP-SYNTH",
        "SWARM-ORCH-DAG",
        "BUILD-KIT-GEN",
        "FIELD-GUIDE-OPS",
        "MAINT-RUN-KIT",
        "SYS1-CLASSIFIER",
        "GOLD-DATA-LABELER",
        "EVAL-BENCH-ARBITER",
        "DECISION-ENVELOPE-GATE",
    ]
    for prompt_id in expected_prompts:
        assert prompt_id in content, f"Missing prompt spec {prompt_id} in {filepath}"


@pytest.mark.parametrize("filepath", [MANUAL_PATH, PROMPT_SUITE_PATH])
def test_layer0_layer1_multiplicative_formulation(filepath: Path):
    """Verify the binary Layer 0 x Layer 1 multiplicative safety mathematics."""
    content = filepath.read_text(encoding="utf-8")
    assert "Layer 0 \\times Layer 1" in content or "Layer 0 Multiplier" in content
    assert "G_{0\\_1} \\times G_{0\\_2} \\times G_{0\\_3} \\times G_{0\\_4}" in content
    assert "COMMIT" in content
    assert "HUMAN_ESCALATE" in content
    assert "BLOCK_AND_LOG" in content


@pytest.mark.parametrize("filepath", [MANUAL_PATH, PROMPT_SUITE_PATH])
def test_epistemic_tags_and_provenance(filepath: Path):
    """Verify epistemic taxonomy tags and bitemporal timestamps are defined."""
    content = filepath.read_text(encoding="utf-8")
    tags = ["[VERIFIED]", "[CALCULATED]", "[MODEL_EST]", "[PENDING]"]
    for tag in tags:
        assert tag in content, f"Missing epistemic tag {tag} in {filepath}"
    assert "as_of_time" in content
    assert "known_at_time" in content


@pytest.mark.parametrize("filepath", [MANUAL_PATH, PROMPT_SUITE_PATH])
def test_self_verification_audit_table(filepath: Path):
    """Verify the live self-verification audit table is fully formatted."""
    content = filepath.read_text(encoding="utf-8")
    assert "Live Self-Verification: Decision Assurance Audit Table" in content
    assert "Pre-Mutation Materiality Gate" in content
    assert "Runtime Constraint Transparency" in content
    assert "Empirical Grounding" in content
    assert "Reasoning & Downside Rigor" in content
    assert "Learning & Reusability" in content
    assert "Final Assured Score" in content
    assert "Routing Verdict: `COMMIT`" in content
