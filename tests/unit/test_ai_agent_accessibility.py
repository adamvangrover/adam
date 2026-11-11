"""Unit tests validating repository accessibility and configuration files for external AI coding agents.

Covers OpenAI Dots, Claude Code, Claude Cowork, and Meta Muse.
"""

import json
from pathlib import Path


def test_openai_instructions_exist_and_valid():
    repo_root = Path(__file__).resolve().parent.parent.parent
    openai_instructions = repo_root / ".openai" / "instructions.md"
    assert openai_instructions.exists(), ".openai/instructions.md must exist"

    content = openai_instructions.read_text(encoding="utf-8")
    assert "AFOS v30.1" in content
    assert "System 1" in content
    assert "System 2" in content
    assert "jsonLogic" in content
    assert "mcp.json" in content
    assert "uv sync" in content


def test_claude_config_exists_and_valid_json():
    repo_root = Path(__file__).resolve().parent.parent.parent
    claude_config = repo_root / ".claude" / "config.json"
    assert claude_config.exists(), ".claude/config.json must exist"

    data = json.loads(claude_config.read_text(encoding="utf-8"))
    assert "project_name" in data
    assert "assistant_integration" in data
    targets = data["assistant_integration"].get("targets", [])
    assert "claude_code" in targets
    assert "claude_cowork" in targets
    assert "anthropic_claude" in targets

    entrypoint_commands = data.get("environment", {}).get("entrypoint_commands", {})
    assert "sync" in entrypoint_commands
    assert "test" in entrypoint_commands
    assert "lint" in entrypoint_commands


def test_claude_md_cowork_and_code_sections():
    repo_root = Path(__file__).resolve().parent.parent.parent
    claude_md = repo_root / "CLAUDE.md"
    assert claude_md.exists(), "CLAUDE.md must exist"

    content = claude_md.read_text(encoding="utf-8")
    assert "Claude Code (CLI)" in content
    assert "Claude Cowork (Multi-Agent Pairing)" in content
    assert "mcp.json" in content


def test_meta_muse_guidelines_exist_and_valid():
    repo_root = Path(__file__).resolve().parent.parent.parent
    muse_guidelines = repo_root / ".meta" / "muse_guidelines.md"
    assert muse_guidelines.exists(), ".meta/muse_guidelines.md must exist"

    content = muse_guidelines.read_text(encoding="utf-8")
    assert "Meta Muse" in content or "Muse Spark" in content
    assert "Thought Compression" in content
    assert "PDIL" in content
    assert "[VERIFIED]" in content


def test_ai_coding_agents_integration_guide_indexed():
    repo_root = Path(__file__).resolve().parent.parent.parent
    guide = repo_root / "docs" / "how-to" / "ai_coding_agents_integration.md"
    assert guide.exists(), "docs/how-to/ai_coding_agents_integration.md must exist"

    guide_content = guide.read_text(encoding="utf-8")
    assert "OpenAI Dots" in guide_content
    assert "Claude Code" in guide_content
    assert "Claude Cowork" in guide_content
    assert "Meta Muse" in guide_content

    # Ensure referenced in docs/how-to/README.md
    how_to_readme = repo_root / "docs" / "how-to" / "README.md"
    assert "ai_coding_agents_integration.md" in how_to_readme.read_text(encoding="utf-8")

    # Ensure referenced in llms.txt
    llms_txt = repo_root / "llms.txt"
    assert "docs/how-to/ai_coding_agents_integration.md" in llms_txt.read_text(encoding="utf-8")


def test_mcp_json_tool_parity():
    repo_root = Path(__file__).resolve().parent.parent.parent
    mcp_file = repo_root / "mcp.json"
    assert mcp_file.exists(), "mcp.json must exist"

    mcp_data = json.loads(mcp_file.read_text(encoding="utf-8"))
    tool_names = {t["name"] for t in mcp_data.get("tools", [])}

    # Verify tools mentioned in assistant guides are registered
    assert "financial_statement_parser" in tool_names
    assert "covenant_headroom_calculator" in tool_names
    assert "arbitration_calculator" in tool_names
