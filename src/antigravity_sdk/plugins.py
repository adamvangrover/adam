"""
Antigravity Plugin Architecture and Registry.
Provides lifecycle management, hook/trigger attachment, and dynamic tool extension.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PluginMetadata:
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "ADAM Team"
    enabled: bool = True
    tags: list[str] = field(default_factory=list)


class AntigravityPlugin:
    """Base class for all Antigravity Plugins."""

    metadata: PluginMetadata

    def __init__(self, name: str, version: str = "1.0.0", description: str = "") -> None:
        self.metadata = PluginMetadata(name=name, version=version, description=description)
        self._tools: dict[str, Callable] = {}
        self._hooks: list[Callable] = []

    def on_load(self) -> None:
        """Called when plugin is loaded into the agent/app runtime."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded or disabled."""
        pass

    def register_tool(self, name: str, func: Callable) -> None:
        """Registers a custom callable as a tool."""
        self._tools[name] = func

    def get_tools(self) -> dict[str, Callable]:
        return dict(self._tools)


class PluginRegistry:
    """Registry managing dynamic plugin discovery, enabling, and invocation."""

    def __init__(self) -> None:
        self._plugins: dict[str, AntigravityPlugin] = {}

    def register(self, plugin: AntigravityPlugin) -> None:
        plugin.on_load()
        self._plugins[plugin.metadata.name] = plugin

    def unregister(self, plugin_name: str) -> Optional[AntigravityPlugin]:
        plugin = self._plugins.pop(plugin_name, None)
        if plugin:
            plugin.on_unload()
        return plugin

    def get_plugin(self, plugin_name: str) -> Optional[AntigravityPlugin]:
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.metadata.name,
                "version": p.metadata.version,
                "description": p.metadata.description,
                "enabled": p.metadata.enabled,
                "tags": p.metadata.tags,
                "tools": list(p.get_tools().keys()),
            }
            for p in self._plugins.values()
        ]

    def get_all_tools(self) -> dict[str, Callable]:
        all_tools: dict[str, Callable] = {}
        for plugin in self._plugins.values():
            if plugin.metadata.enabled:
                all_tools.update(plugin.get_tools())
        return all_tools


# Global default registry
default_plugin_registry = PluginRegistry()


def plugin(name: str, version: str = "1.0.0", description: str = ""):
    """Decorator to declare a simple functional plugin."""
    def decorator(cls_or_func):
        if inspect.isclass(cls_or_func) and issubclass(cls_or_func, AntigravityPlugin):
            inst = cls_or_func()
            default_plugin_registry.register(inst)
            return cls_or_func
        # Otherwise wrap function as a tool inside a generated plugin
        p = AntigravityPlugin(name=name, version=version, description=description)
        p.register_tool(name, cls_or_func)
        default_plugin_registry.register(p)
        return cls_or_func
    return decorator
