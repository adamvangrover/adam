"""
Tool registration and in-process execution engine for Antigravity SDK.
"""

import inspect
import logging
from typing import Any, Callable, Optional
from .types import ToolCall

logger = logging.getLogger(__name__)


class ToolRunner:
    """
    Manages in-process tool registration, schema extraction, and invocation dispatch.
    """

    def __init__(self, tools: Optional[list[Callable[..., Any]]] = None) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)

    def register_tool(self, fn: Callable[..., Any], name: Optional[str] = None) -> None:
        """Registers a Python function as an executable tool."""
        tool_name = name or fn.__name__
        self._registry[tool_name] = fn
        logger.debug("Registered tool: %s", tool_name)

    def get_tool_names(self) -> list[str]:
        """Returns the list of registered tool names."""
        return list(self._registry.keys())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Generates JSON schema descriptions for all registered tools."""
        schemas = []
        for name, fn in self._registry.items():
            doc = inspect.getdoc(fn) or ""
            sig = inspect.signature(fn)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                param_type = "string"
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is dict:
                    param_type = "object"
                elif param.annotation is list:
                    param_type = "array"

                properties[param_name] = {"type": param_type}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            schemas.append({
                "name": name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return schemas

    async def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """Executes a ToolCall dispatch asynchronously with signature matching."""
        if tool_call.name not in self._registry:
            raise ValueError(f"Tool '{tool_call.name}' is not registered.")

        fn = self._registry[tool_call.name]
        provided_args = tool_call.arguments or {}

        sig = inspect.signature(fn)
        bound_args = {}
        for param_name, param in sig.parameters.items():
            if param_name in provided_args:
                bound_args[param_name] = provided_args[param_name]
            elif param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default

        if inspect.iscoroutinefunction(fn):
            return await fn(**bound_args)
        else:
            return fn(**bound_args)
