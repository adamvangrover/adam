"""
Jules Helper Ecosystem: CLI, Web IDE, and Python SDK.
Self-contained, local-first agent framework for ADAM OS.
"""

from src.jules.sdk import (
    BaseLLMProvider,
    JulesAgent,
    JulesConfig,
    JulesEngine,
    JulesMemoryManager,
    JulesToolRegistry,
    get_llm_provider,
)

__version__ = "1.0.0"

__all__ = [
    "BaseLLMProvider",
    "JulesAgent",
    "JulesConfig",
    "JulesEngine",
    "JulesMemoryManager",
    "JulesToolRegistry",
    "get_llm_provider",
]
