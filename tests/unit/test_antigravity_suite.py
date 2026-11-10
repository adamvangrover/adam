"""
Comprehensive Unit & Integration Test Suite for Antigravity Suite:
- Antigravity SDK
- Antigravity CLI
- Antigravity IDE (FastAPI handlers & router logic)
- Antigravity Extensions & Plugin Manifests
"""

import json
from pathlib import Path

import pytest

from src.antigravity_cli import (
    build_parser,
    cmd_extensions,
    cmd_status,
    cmd_tools,
    cmd_triggers,
    main as cli_main,
)
from src.antigravity_ide.server import (
    ChatRequest,
    FileSaveRequest,
    TaskRequest,
    api_chat,
    api_list_extensions,
    api_list_files,
    api_list_tools,
    api_list_triggers,
    api_plan,
    api_read_file,
    api_run,
    api_save_file,
    get_status,
)
from src.antigravity_sdk import (
    Agent,
    CapabilitiesConfig,
    HookRunner,
    LocalAgentConfig,
    allow,
    ask_user,
    deny,
    enforce,
    every,
)


# =========================================================================
# 1. Antigravity SDK Tests
# =========================================================================

@pytest.mark.asyncio
async def test_antigravity_sdk_agent_and_tools() -> None:
    def add(a: int, b: int) -> int:
        """Adds two integers."""
        return a + b

    config = LocalAgentConfig(
        tools=[add],
        capabilities=CapabilitiesConfig(allow_code_execution=True),
    )
    async with Agent(config) as agent:
        resp = await agent.chat("Compute 2 + 2")
        text = await resp.text()
        assert "Processed prompt turn 1" in text
        assert "Compute 2 + 2" in text


@pytest.mark.asyncio
async def test_antigravity_sdk_hooks_and_triggers() -> None:
    policies = [
        allow("read_*"),
        deny("delete_*"),
        ask_user("transfer_*"),
        enforce("safety_invariants"),
    ]
    runner = HookRunner(policies=policies)
    res_allow = runner.evaluate_tool_call("read_file", {"filepath": "foo.py"})
    assert res_allow.action == "ALLOW"

    res_deny = runner.evaluate_tool_call("delete_database", {})
    assert res_deny.action == "DENY"

    res_ask = runner.evaluate_tool_call("transfer_funds", {"amount": 100})
    assert res_ask.action == "ASK_USER"

    # Test trigger helper
    trigger = every(10.0, lambda ctx: None)
    assert trigger.interval_seconds == 10.0


# =========================================================================
# 2. Antigravity CLI Tests
# =========================================================================

def test_antigravity_cli_parser() -> None:
    parser = build_parser()
    args_run = parser.parse_args(["run", "Audit task objective"])
    assert args_run.command == "run"
    assert args_run.prompt == "Audit task objective"

    args_status = parser.parse_args(["status"])
    assert args_status.command == "status"

    args_serve = parser.parse_args(["serve", "--port", "8088"])
    assert args_serve.command == "serve"
    assert args_serve.port == 8088


def test_antigravity_cli_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cmd_tools() == 0
    captured = capsys.readouterr()
    assert "Registered Antigravity Tools" in captured.out

    assert cmd_triggers() == 0
    captured = capsys.readouterr()
    assert "Antigravity Background Triggers" in captured.out

    assert cmd_status() == 0
    captured = capsys.readouterr()
    assert "Antigravity Runtime & Environment Diagnostics" in captured.out

    parser = build_parser()
    ext_args = parser.parse_args(["extensions", "--dir", "extensions"])
    assert cmd_extensions(ext_args) == 0
    captured = capsys.readouterr()
    assert "antigravity-ide-extension" in captured.out


def test_antigravity_cli_main_entry() -> None:
    rc = cli_main(["status"])
    assert rc == 0


# =========================================================================
# 3. Antigravity IDE FastAPI Backend Handlers Tests
# =========================================================================

def test_antigravity_ide_status_handler() -> None:
    data = get_status()
    assert data["status"] == "online"
    assert "capabilities" in data
    assert data["capabilities"]["file_system_write"] is True


@pytest.mark.asyncio
async def test_antigravity_ide_chat_and_plan_handlers() -> None:
    chat_resp = await api_chat(ChatRequest(message="Audit repo status", model="gemini-2.5-pro"))
    assert "reply" in chat_resp
    assert "session_id" in chat_resp

    plan_resp = await api_plan(TaskRequest(prompt="Deploy credit models"))
    assert "plan" in plan_resp
    assert len(plan_resp["plan"]) >= 4

    run_resp = await api_run(TaskRequest(prompt="Run repository audit"))
    assert run_resp["status"] == "SUCCESS"
    assert len(run_resp["prov_o_audit_trail"]) == 2


def test_antigravity_ide_file_handlers(tmp_path: Path) -> None:
    # Test file list
    list_data = api_list_files(path=".")
    assert "entries" in list_data

    # Test file save and read
    test_file = tmp_path / "test_scratchpad.py"
    save_resp = api_save_file(FileSaveRequest(filepath=str(test_file), content="print('hello antigravity')"))
    assert save_resp["status"] == "saved"

    read_resp = api_read_file(filepath=str(test_file))
    assert read_resp["content"] == "print('hello antigravity')"


def test_antigravity_ide_tools_triggers_extensions_handlers() -> None:
    tools_resp = api_list_tools()
    assert "tools" in tools_resp
    assert len(tools_resp["tools"]) > 0

    triggers_resp = api_list_triggers()
    assert "active_triggers" in triggers_resp

    ext_resp = api_list_extensions()
    assert "extensions" in ext_resp
    assert len(ext_resp["extensions"]) > 0
    assert "plugins" in ext_resp
    assert len(ext_resp["plugins"]) > 0


# =========================================================================
# 4. Antigravity Extensions & Plugin Manifests Tests
# =========================================================================

def test_antigravity_extension_manifest() -> None:
    pkg_file = Path("extensions/antigravity_extensions/package.json")
    assert pkg_file.exists()
    pkg_data = json.loads(pkg_file.read_text(encoding="utf-8"))
    assert pkg_data["name"] == "antigravity-ide-extension"
    assert "contributes" in pkg_data
    assert len(pkg_data["contributes"]["commands"]) >= 3


def test_antigravity_plugins_manifest() -> None:
    sentinel = Path("plugins/sentinel_plugin.json")
    assert sentinel.exists()
    s_data = json.loads(sentinel.read_text(encoding="utf-8"))
    assert s_data["name"] == "sentinel-security-plugin"
    assert s_data["zero_trust"]["require_ecdsa_signature"] is True

    prov_o = Path("plugins/prov_o_plugin.json")
    assert prov_o.exists()
    p_data = json.loads(prov_o.read_text(encoding="utf-8"))
    assert p_data["name"] == "w3c-prov-o-plugin"
    assert p_data["hash_algorithm"] == "SHA-256"
