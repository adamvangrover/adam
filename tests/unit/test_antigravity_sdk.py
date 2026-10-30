"""
Unit tests for Antigravity SDK and Jules Replacement Agent.
"""

import asyncio
import pytest
from pathlib import Path

from src.antigravity_sdk import (
    Agent,
    CapabilitiesConfig,
    Conversation,
    Document,
    HookRunner,
    Image,
    JulesAgent,
    LiteRTAgentConfig,
    LocalAgentConfig,
    McpStdioServer,
    ToolRunner,
    allow,
    deny,
    every,
    from_file,
)


@pytest.mark.asyncio
async def test_agent_async_context_manager_and_chat():
    config = LocalAgentConfig(system_instructions="Test agent instructions")
    async with Agent(config) as agent:
        response = await agent.chat("What files are in the repository?")
        text = await response.text()
        assert "Processed prompt turn 1" in text
        assert "What files are in the repository?" in text


@pytest.mark.asyncio
async def test_streaming_tokens_and_thoughts():
    config = LocalAgentConfig()
    async with Agent(config) as agent:
        response = await agent.chat("Generate code snippet")

        tokens = []
        async for token in response:
            tokens.append(token)
        assert len(tokens) > 0

        thoughts = []
        async for thought in response.thoughts:
            thoughts.append(thought)
        assert len(thoughts) > 0


@pytest.mark.asyncio
async def test_tool_runner_registration_and_execution():
    def get_weather(city: str) -> str:
        """Returns the weather for a city."""
        return f"Sunny in {city}"

    tool_runner = ToolRunner(tools=[get_weather])
    assert "get_weather" in tool_runner.get_tool_names()

    schemas = tool_runner.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "get_weather"

    config = LocalAgentConfig(tools=[get_weather])
    async with Agent(config) as agent:
        response = await agent.chat("Check weather in Tokyo")
        text = await response.text()
        assert "Tokyo" in text


@pytest.mark.asyncio
async def test_policy_hooks_enforcement():
    def execute_cmd(cmd: str) -> str:
        return f"Executed: {cmd}"

    policies = [
        allow("*"),
        deny("execute_cmd"),
    ]
    hook_runner = HookRunner(policies=policies)
    policy = hook_runner.evaluate_tool_call("execute_cmd", {"cmd": "ls"})
    assert policy.action == "DENY"


@pytest.mark.asyncio
async def test_multimodal_file_helpers(tmp_path: Path):
    test_file = tmp_path / "spec.pdf"
    test_file.write_bytes(b"PDF header content")

    doc = from_file(test_file)
    assert isinstance(doc, Document)
    assert doc.mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_jules_agent_replacement_workflow():
    jules = JulesAgent()
    res = await jules.execute_task("Perform repository integrity check")

    assert res["status"] == "SUCCESS"
    assert "Processed prompt turn 1" in res["response"]
    assert len(res["prov_o_audit_trail"]) == 2
    assert res["prov_o_audit_trail"][0]["prov:Activity"] == "TaskStart"
    assert res["prov_o_audit_trail"][1]["prov:Activity"] == "TaskComplete"
