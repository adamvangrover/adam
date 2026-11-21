"""
Unit tests for AFOS v30.1 Repository-as-a-Prompt Runner for Google Antigravity SDK.
"""

from __future__ import annotations

import pytest
import scripts.run_antigravity_repo_prompt as runner


def test_build_repo_system_prompt():
    """Verify system prompt hydrates repo manifests correctly."""
    prompt = runner.build_repo_system_prompt(include_full_knowledge_base=True)
    assert "# SYSTEM BOOT: ADAM v30.1" in prompt
    assert "LLM_README.md" in prompt
    assert "AGENTS.md" in prompt

    prompt_short = runner.build_repo_system_prompt(include_full_knowledge_base=False)
    assert "# SYSTEM BOOT: ADAM v30.1" in prompt_short


def test_infer_routing_path():
    """Verify routing path inference across workflow, swarm, and default DAG modes."""
    assert runner._infer_routing_path("any query", mode="workflow") == "SYSTEM_2_WORKFLOW"
    assert runner._infer_routing_path("run swarm scan across feeds", mode="chat") == "SYSTEM_1_SWARM"
    assert runner._infer_routing_path("surveillance telemetry", mode="chat") == "SYSTEM_1_SWARM"
    assert runner._infer_routing_path("hive pulse", mode="chat") == "SYSTEM_1_SWARM"
    assert runner._infer_routing_path("calculate dscr waterfall", mode="chat") == "SYSTEM_2_DAG"


def test_build_prov_o_envelope():
    """Verify W3C PROV-O envelope building and grounding checks."""
    envelope = runner.build_prov_o_envelope(
        query="Explain LMT contagion gate",
        answer="Analysis of LMT contagion dynamics.",
        routing_path="SYSTEM_2_DAG",
        source_data_object="[docs/AGENTS_KNOWLEDGE_BASE.md:v30.1]",
        confidence_score=0.95,
    )
    assert envelope["w3c_prov_o_grounded"] is True
    assert "state_sha256" in envelope
    assert envelope["agent_output"]["confidence"] == 0.95
    assert "[docs/AGENTS_KNOWLEDGE_BASE.md:v30.1]" in envelope["agent_output"]["sources"]
    assert "[docs/AGENTS_KNOWLEDGE_BASE.md:v30.1]" in envelope["agent_output"]["answer"]


def test_build_prov_o_envelope_grounding_violation():
    """Verify grounding error when source data object is empty."""
    with pytest.raises(ValueError, match="W3C PROV-O Grounding Violation"):
        runner.build_prov_o_envelope(
            query="test",
            answer="test",
            source_data_object="",
        )


def test_run_offline_deterministic_fallback():
    """Verify offline deterministic fallback returns valid envelope."""
    envelope = runner._run_offline_deterministic_fallback(
        query="scan liquidity indicators",
        reason="test offline execution",
        mode="chat",
    )
    assert envelope["w3c_prov_o_grounded"] is True
    assert envelope["agent_output"]["metadata"]["routing_path"] == "SYSTEM_1_SWARM"
    assert "test offline execution" in envelope["agent_output"]["answer"]


def test_parse_args():
    """Verify CLI argument parsing."""
    args = runner.parse_args(["--mode", "workflow", "--runtime", "litert", "--vertex", "--write", "Audit query"])
    assert args.mode == "workflow"
    assert args.runtime == "litert"
    assert args.vertex is True
    assert args.write is True
    assert args.query == "Audit query"

    default_args = runner.parse_args([])
    assert default_args.mode == "chat"
    assert default_args.runtime == "gemini"
    assert default_args.vertex is False
    assert default_args.write is False


@pytest.mark.asyncio
async def test_run_with_antigravity_sdk_fallback():
    """Verify run_with_antigravity_sdk executes graceful offline fallback when SDK is absent."""
    envelope = await runner.run_with_antigravity_sdk(
        query="Verify grounding for sovereign bundle",
        mode="chat",
        runtime="gemini",
    )
    assert envelope["w3c_prov_o_grounded"] is True
    assert envelope["agent_output"]["metadata"]["routing_path"] == "SYSTEM_2_DAG"
