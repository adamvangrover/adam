from typing import Dict, Type
from .base_prompt_plugin import BasePromptPlugin
from .plugins.financial_truth_plugin import FinancialTruthPlugin
from .plugins.tree_of_thoughts_plugin import TreeOfThoughtsPlugin
from .plugins.chain_of_verification_plugin import ChainOfVerificationPlugin
# from .plugins.example_plugin import ExamplePlugin # Example plugin available for reference but not registered by default


class PromptRegistry:
    """
    Registry for managing prompt plugins.
    Allows registering plugins by ID and retrieving them.
    """
    _registry: Dict[str, Type[BasePromptPlugin]] = {}

    @classmethod
    def register(cls, plugin_cls: Type[BasePromptPlugin]):
        """Registers a plugin class. Uses the class name."""
        cls._registry[plugin_cls.__name__] = plugin_cls

    @classmethod
    def get(cls, plugin_name: str) -> Type[BasePromptPlugin]:
        if plugin_name not in cls._registry:
            raise ValueError(f"Plugin {plugin_name} not found in registry.")
        return cls._registry[plugin_name]

    @classmethod
    def list_plugins(cls) -> list[str]:
        return list(cls._registry.keys())


# Register default plugins
PromptRegistry.register(FinancialTruthPlugin)
PromptRegistry.register(TreeOfThoughtsPlugin)
PromptRegistry.register(ChainOfVerificationPlugin)
# PromptRegistry.register(ExamplePlugin)
