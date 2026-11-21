"""
Unit & Integration Tests for Jules Python SDK.
"""

from pathlib import Path

from src.jules.sdk.agent import JulesAgent
from src.jules.sdk.config import JulesConfig
from src.jules.sdk.memory import JulesMemoryManager
from src.jules.sdk.tools import JulesToolRegistry


def test_jules_config_defaults() -> None:
    cfg = JulesConfig()
    assert cfg.provider in ["mock", "ollama", "openai", "anthropic"]
    assert cfg.workspace_root == str(Path.cwd())
    assert cfg.max_turns == 15


def test_jules_tool_registry(tmp_path: Path) -> None:
    registry = JulesToolRegistry(workspace_root=str(tmp_path))
    tools = registry.list_tools()
    tool_names = [t["name"] for t in tools]

    assert "list_files" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "delete_file" in tool_names
    assert "run_bash" in tool_names

    # Test file write & read
    write_res = registry.execute_tool("write_file", {"filepath": "sample.txt", "content": "Hello Jules"})
    assert write_res["status"] == "success"

    read_res = registry.execute_tool("read_file", {"filepath": "sample.txt"})
    assert read_res["status"] == "success"
    assert read_res["result"] == "Hello Jules"

    # Test audit log
    audit_log = registry.get_audit_log()
    assert len(audit_log) >= 2
    assert "context_provenance" in audit_log[0]


def test_jules_agent_chat_and_run(tmp_path: Path) -> None:
    cfg = JulesConfig(provider="mock", workspace_root=str(tmp_path))
    agent = JulesAgent(config=cfg)

    # Test chat
    reply = agent.chat("How are you?")
    assert "mock-jules-v1" in reply or "analyzed" in reply.lower() or "jules" in reply.lower()

    # Test set_plan
    plan = agent.set_plan("Create sample module")
    assert isinstance(plan, list)
    assert len(plan) > 0

    # Test run
    task_res = agent.run("Check repo files")
    assert task_res["status"] == "completed"
    assert "plan" in task_res
    assert "audit_log" in task_res
    assert "context_provenance" in task_res


def test_jules_memory_manager(tmp_path: Path) -> None:
    mem = JulesMemoryManager(workspace_root=str(tmp_path))
    mem.save_memory("key1", "val1")
    memories = mem.load_memories()
    assert "key1" in memories
    assert memories["key1"]["value"] == "val1"

    mem.log_task_execution("prompt1", ["step1"], "completed", "ok")
    history = mem.get_history(limit=5)
    assert len(history) == 1
    assert history[0]["prompt"] == "prompt1"
