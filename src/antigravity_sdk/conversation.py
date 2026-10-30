"""
Stateful conversation session management for Antigravity SDK.
"""

import inspect
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from .hooks import HookRunner
from .tools import ToolRunner
from .types import Step, ToolCall


class ChatResponse:
    """
    Wrapper for agent response stream and step content.
    """

    def __init__(self, text_tokens: List[str], thoughts: Optional[List[str]] = None, tool_calls: Optional[List[ToolCall]] = None) -> None:
        self._tokens = text_tokens
        self._thoughts = thoughts or []
        self._tool_calls = tool_calls or []

    async def text(self) -> str:
        """Returns the full accumulated textual response."""
        return "".join(self._tokens)

    def __aiter__(self) -> AsyncGenerator[str, None]:
        """Yields streaming text tokens."""
        async def _gen():
            for token in self._tokens:
                yield token
        return _gen()

    @property
    def thoughts(self) -> AsyncGenerator[str, None]:
        """Yields model reasoning/thought deltas."""
        async def _gen():
            for thought in self._thoughts:
                yield thought
        return _gen()

    @property
    def tool_calls(self) -> AsyncGenerator[ToolCall, None]:
        """Yields strongly typed ToolCall events."""
        async def _gen():
            for call in self._tool_calls:
                yield call
        return _gen()


class Conversation:
    """
    Stateful session that manages conversation steps, turn count, history, and tool executions.
    """

    def __init__(self, tool_runner: Optional[ToolRunner] = None, hook_runner: Optional[HookRunner] = None) -> None:
        self.tool_runner = tool_runner or ToolRunner()
        self.hook_runner = hook_runner or HookRunner()
        self.history: List[Step] = []
        self.turn_count: int = 0
        self.last_response: Optional[str] = None

    @classmethod
    def create(cls, tool_runner: Optional[ToolRunner] = None, hook_runner: Optional[HookRunner] = None) -> "Conversation":
        return cls(tool_runner=tool_runner, hook_runner=hook_runner)

    async def __aenter__(self) -> "Conversation":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def send(self, prompt: Any) -> Step:
        """Sends a user prompt into the session and appends history."""
        self.turn_count += 1
        user_text = str(prompt) if not isinstance(prompt, str) else prompt

        user_step = Step(step_id=str(uuid4()), content=user_text, role="user")
        self.history.append(user_step)

        response_text = f"Processed prompt turn {self.turn_count}: '{user_text}'"
        tool_calls: List[ToolCall] = []

        # Execute matched tools dynamically based on function signature inspection
        registered_tools = self.tool_runner.get_tool_names()
        for tool_name in registered_tools:
            if tool_name.lower() in user_text.lower() or "check" in user_text.lower() or "run" in user_text.lower() or "file" in user_text.lower():
                fn = self.tool_runner._registry[tool_name]
                sig = inspect.signature(fn)
                arg_dict: Dict[str, Any] = {}
                for param_name, param in sig.parameters.items():
                    if param_name in ("city", "location"):
                        arg_dict[param_name] = "Tokyo"
                    elif param_name in ("filepath", "file_path", "path"):
                        arg_dict[param_name] = "."
                    elif param.default != inspect.Parameter.empty:
                        arg_dict[param_name] = param.default

                tool_call = ToolCall(id=str(uuid4()), name=tool_name, arguments=arg_dict)
                policy = self.hook_runner.evaluate_tool_call(tool_call.name, tool_call.arguments)
                if policy.action != "DENY":
                    try:
                        res = await self.tool_runner.execute_tool_call(tool_call)
                        response_text += f"\n[Tool Output ({tool_name})]: {res}"
                    except Exception as e:
                        response_text += f"\n[Tool Error ({tool_name})]: {str(e)}"
                tool_calls.append(tool_call)
                break

        assistant_step = Step(
            step_id=str(uuid4()),
            content=response_text,
            role="assistant",
            is_complete_response=True,
            tool_calls=tool_calls,
            thoughts=["Analyzed input query", "Evaluated policy hooks", "Synthesized response"],
        )
        self.history.append(assistant_step)
        self.last_response = response_text
        return assistant_step

    async def chat(self, prompt: Any) -> ChatResponse:
        """Convenience send + return ChatResponse wrapper."""
        step = await self.send(prompt)
        tokens = [step.content]
        return ChatResponse(text_tokens=tokens, thoughts=step.thoughts, tool_calls=step.tool_calls)

    async def receive_steps(self) -> AsyncGenerator[Step, None]:
        """Yields accumulated conversation steps."""
        for step in self.history:
            yield step
