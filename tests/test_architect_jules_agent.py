"""
Unit tests for AFOS v30.1 Architect Agent (Jules) — `architect_jules`
and `@google/jules` CLI command builder (`core/agents/architect_agent.py`).
Validates both successful and adversarial execution paths per Invariant J.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.agents.architect_agent import (
    ArchitectAgent,
    JulesCommandBuilder,
    JulesSessionRequest,
)


def test_jules_remote_new_command_builder():
    req = JulesSessionRequest(
        subcommand="remote_new",
        repo="torvalds/linux",
        session_prompt="write unit tests",
        parallel=3,
    )
    args = JulesCommandBuilder.build_args(req)
    assert args == [
        "jules",
        "remote",
        "new",
        "--repo",
        "torvalds/linux",
        "--session",
        "write unit tests",
        "--parallel",
        "3",
    ]
    cli_str = JulesCommandBuilder.format_cli_string(req)
    assert "jules remote new --repo torvalds/linux --session 'write unit tests' --parallel 3" == cli_str


def test_jules_remote_pull_and_list_commands():
    pull_req = JulesSessionRequest(subcommand="remote_pull", session_id="123456")
    assert JulesCommandBuilder.build_args(pull_req) == [
        "jules",
        "remote",
        "pull",
        "--session",
        "123456",
    ]

    list_repo_req = JulesSessionRequest(subcommand="remote_list_repo")
    assert JulesCommandBuilder.build_args(list_repo_req) == [
        "jules",
        "remote",
        "list",
        "--repo",
    ]

    list_session_req = JulesSessionRequest(subcommand="remote_list_session")
    assert JulesCommandBuilder.build_args(list_session_req) == [
        "jules",
        "remote",
        "list",
        "--session",
    ]


def test_jules_global_theme_and_completion_commands():
    tui_req = JulesSessionRequest(subcommand="tui", theme="light")
    assert JulesCommandBuilder.build_args(tui_req) == ["jules", "--theme", "light"]

    comp_req = JulesSessionRequest(subcommand="completion", shell="bash")
    assert JulesCommandBuilder.build_args(comp_req) == ["jules", "completion", "bash"]


def test_adversarial_repo_and_session_id_rejected():
    with pytest.raises(ValidationError):
        JulesSessionRequest(
            subcommand="remote_new",
            repo="adamvangrover/adam; rm -rf /",
            session_prompt="test",
        )

    with pytest.raises(ValidationError):
        JulesSessionRequest(
            subcommand="remote_pull",
            session_id="123456 && cat /etc/passwd",
        )

    with pytest.raises(ValidationError):
        JulesSessionRequest(
            subcommand="remote_new",
            repo="adamvangrover/adam",
            session_prompt="   ",
        )


def test_architect_agent_code_audit_and_prov_o_trace():
    agent = ArchitectAgent()

    clean_code = "from pydantic import BaseModel\nclass RiskSchema(BaseModel):\n    pd: float\n"
    clean_audit = agent.audit_code_snippet(clean_code, module_path="core/agents/risk.py")
    assert clean_audit["passed"] is True
    assert clean_audit["frontend_sync_status"] == "decoupled"

    adversarial_code = (
        "import streamlit as st\n"
        "from experimental.proto import helper\n"
        "data = pickle.load(open('x.pkl', 'rb'))\n"
    )
    bad_audit = agent.audit_code_snippet(adversarial_code, module_path="core/agents/bad.py")
    assert bad_audit["passed"] is False
    assert bad_audit["frontend_sync_status"] == "violation_detected"
    assert len(bad_audit["violations"]) == 3

    # Verify W3C PROV-O telemetry generation
    agent_clean = ArchitectAgent()
    req = JulesSessionRequest(
        subcommand="remote_new",
        repo="adamvangrover/adam",
        session_prompt="Audit core/ for P0 security compliance",
        parallel=2,
    )
    out = agent_clean.prepare_jules_session(req, execute=False)
    assert out["confidence"] >= 0.85
    assert len(out["prov_o_audit_trail"]) == 1
    assert out["provenance_trace"]["jsonLogic_version"] == "2.0.0"
    assert len(out["provenance_trace"]["content_hash"]) == 64

