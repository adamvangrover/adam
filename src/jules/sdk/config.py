"""
Jules SDK Configuration Module.
Provides workspace, model, provider, and runtime safety settings for Jules Agent.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class JulesConfig(BaseModel):
    """
    Configuration options for Jules Agent & Engine.
    """
    provider: str = Field(
        default_factory=lambda: os.getenv("JULES_PROVIDER", "mock"),
        description="LLM provider: 'mock', 'ollama', 'openai', 'anthropic', 'litellm'"
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("JULES_MODEL", "mock-jules-v1"),
        description="Model name/identifier"
    )
    workspace_root: str = Field(
        default_factory=lambda: str(Path.cwd()),
        description="Root path of the working workspace"
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature"
    )
    max_turns: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Maximum turns per execution loop"
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("JULES_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        description="Optional API key for cloud providers"
    )
    ollama_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"),
        description="Ollama API base URL"
    )
    enable_prov_o_audit: bool = Field(
        default=True,
        description="Enforce W3C PROV-O audit logging for all tool executions"
    )
