"""
Configuration objects for Antigravity SDK agents.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from .types import McpStdioServer


@dataclass
class CapabilitiesConfig:
    """Capabilities and tool permissions configuration for agents."""
    read_only: bool = False
    allow_code_execution: bool = True
    allow_terminal: bool = True
    allow_file_system_write: bool = True


@dataclass
class LocalAgentConfig:
    """Configuration for local or remote API-backed Antigravity agents."""
    system_instructions: Optional[str] = None
    api_key: Optional[str] = None
    vertex: bool = False
    project: Optional[str] = None
    location: Optional[str] = None
    tools: list[Callable[..., Any]] = field(default_factory=list)
    mcp_servers: list[McpStdioServer] = field(default_factory=list)
    capabilities: CapabilitiesConfig = field(default_factory=CapabilitiesConfig)
    policies: list[Any] = field(default_factory=list)
    triggers: list[Any] = field(default_factory=list)
    model_name: str = "gemini-2.5-pro"


@dataclass
class LiteRTAgentConfig:
    """Configuration for offline, local GPU/CPU LiteRT-LM models (e.g. Gemma 4)."""
    model_path: str
    system_instructions: Optional[str] = None
    tools: list[Callable[..., Any]] = field(default_factory=list)
    capabilities: CapabilitiesConfig = field(default_factory=CapabilitiesConfig)
    context_size: int = 65536

    def lightweight(self) -> "LiteRTAgentConfig":
        """Returns a lightweight configuration tuned for fast local inference."""
        self.context_size = 32768
        return self
