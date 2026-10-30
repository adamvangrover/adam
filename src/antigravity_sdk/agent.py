"""
High-level Agent entry point and async context manager for Antigravity SDK.
"""

from typing import Any

from .config import LiteRTAgentConfig, LocalAgentConfig
from .conversation import ChatResponse, Conversation
from .hooks import HookRunner
from .tools import ToolRunner
from .triggers import TriggerRunner


class Agent:
    """
    High-level, batteries-included agent manager interface.
    """

    def __init__(self, config: LocalAgentConfig | LiteRTAgentConfig | None = None) -> None:
        self.config = config or LocalAgentConfig()
        tools = getattr(self.config, "tools", [])
        policies = getattr(self.config, "policies", [])
        triggers = getattr(self.config, "triggers", [])

        self.tool_runner = ToolRunner(tools=tools)
        self.hook_runner = HookRunner(policies=policies)
        self.trigger_runner = TriggerRunner(triggers=triggers)
        self.conversation = Conversation(tool_runner=self.tool_runner, hook_runner=self.hook_runner)

    async def __aenter__(self) -> "Agent":
        """Starts background triggers and initializes conversation session."""
        await self.trigger_runner.start_all(agent=self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stops active background triggers upon exit."""
        self.trigger_runner.stop_all()

    async def chat(self, prompt: Any) -> ChatResponse:
        """Sends prompt to agent and returns streamable ChatResponse."""
        return await self.conversation.chat(prompt)
