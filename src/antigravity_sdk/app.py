"""
Antigravity Application & Workbench Framework.
Orchestrates agents, plugins, extensions, runtimes, and local tool executions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .extensions import ExtensionHost, default_extension_host
from .plugins import PluginRegistry, default_plugin_registry
from .runtimes import default_runtime_registry
from .tools import ToolRunner

logger = logging.getLogger("antigravity.app")


@dataclass
class AppConfig:
    workspace_root: str = "."
    runtime_mode: str = "offline"
    enable_plugins: bool = True
    enable_extensions: bool = True
    auto_discovery: bool = True


class AntigravityApp:
    """Unified Antigravity Application & Workbench Controller."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.workspace_root = Path(self.config.workspace_root).resolve()
        self.plugins: PluginRegistry = default_plugin_registry
        self.extensions: ExtensionHost = default_extension_host
        self.runtimes = default_runtime_registry
        self.runtime = self.runtimes.get_runtime(self.config.runtime_mode)
        self.tool_runner = ToolRunner()
        self._audit_log: list[dict[str, Any]] = []

    def log_event(self, event_type: str, details: dict[str, Any]) -> None:
        entry = {
            "event": event_type,
            "details": details,
        }
        self._audit_log.append(entry)

    def execute_prompt(self, prompt: str) -> dict[str, Any]:
        """Runs prompt through current runtime and dispatches plugin/extension hooks."""
        self.log_event("execute_prompt", {"prompt": prompt})
        resp = self.runtime.generate(prompt)
        return {
            "status": "success",
            "runtime": self.config.runtime_mode,
            "response": resp,
            "active_plugins": len(self.plugins.list_plugins()),
            "active_extensions": len(self.extensions.list_extensions()),
        }

    def run_tool(self, name: str, params: dict[str, Any]) -> Any:
        # Check plugin tools first
        plugin_tools = self.plugins.get_all_tools()
        if name in plugin_tools:
            res = plugin_tools[name](**params)
            self.log_event("tool_call_plugin", {"name": name, "params": params})
            return res
        return {"error": f"Tool '{name}' not found."}

    def get_status(self) -> dict[str, Any]:
        return {
            "app": "Antigravity App & Workbench",
            "workspace": str(self.workspace_root),
            "runtime_mode": self.config.runtime_mode,
            "plugins": self.plugins.list_plugins(),
            "extensions": self.extensions.list_extensions(),
            "runtimes": self.runtimes.list_runtimes(),
            "audit_events": len(self._audit_log),
        }
