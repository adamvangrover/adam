"""
Antigravity Extensions System.
Provides modular extensions for IDE, CLI, and App (syntax highlighters, linters, diagnostics, commands).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ExtensionManifest:
    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "ADAM Ecosystem"
    contributes: dict[str, Any] = field(default_factory=dict)
    active: bool = True


class AntigravityExtension:
    """Base class for Antigravity Extensions."""

    def __init__(self, manifest: ExtensionManifest) -> None:
        self.manifest = manifest
        self._commands: dict[str, Callable] = {}
        self._diagnostics_providers: list[Callable] = []

    def activate(self) -> None:
        """Lifecycle hook when extension is activated."""
        pass

    def deactivate(self) -> None:
        """Lifecycle hook when extension is deactivated."""
        pass

    def register_command(self, command_id: str, handler: Callable) -> None:
        self._commands[command_id] = handler

    def execute_command(self, command_id: str, *args, **kwargs) -> Any:
        if command_id not in self._commands:
            raise KeyError(f"Command '{command_id}' not found in extension {self.manifest.id}")
        return self._commands[command_id](*args, **kwargs)

    def register_diagnostics(self, provider: Callable) -> None:
        self._diagnostics_providers.append(provider)

    def run_diagnostics(self, code: str, filepath: str) -> list[dict[str, Any]]:
        results = []
        for provider in self._diagnostics_providers:
            try:
                res = provider(code, filepath)
                if isinstance(res, list):
                    results.extend(res)
                elif res:
                    results.append(res)
            except Exception as e:
                results.append({"error": str(e), "provider": getattr(provider, "__name__", "unknown")})
        return results


class ExtensionHost:
    """Extension Host managing lifecycle and command dispatch."""

    def __init__(self) -> None:
        self._extensions: dict[str, AntigravityExtension] = {}

    def install(self, extension: AntigravityExtension) -> None:
        extension.activate()
        self._extensions[extension.manifest.id] = extension

    def uninstall(self, extension_id: str) -> Optional[AntigravityExtension]:
        ext = self._extensions.pop(extension_id, None)
        if ext:
            ext.deactivate()
        return ext

    def get_extension(self, extension_id: str) -> Optional[AntigravityExtension]:
        return self._extensions.get(extension_id)

    def list_extensions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": ext.manifest.id,
                "name": ext.manifest.name,
                "version": ext.manifest.version,
                "description": ext.manifest.description,
                "active": ext.manifest.active,
                "contributes": ext.manifest.contributes,
                "commands": list(ext._commands.keys()),
            }
            for ext in self._extensions.values()
        ]

    def dispatch_command(self, command_id: str, *args, **kwargs) -> Any:
        for ext in self._extensions.values():
            if command_id in ext._commands:
                return ext.execute_command(command_id, *args, **kwargs)
        raise KeyError(f"No extension registered command '{command_id}'")


default_extension_host = ExtensionHost()
