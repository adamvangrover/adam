import importlib
import pkgutil
import sys
import os
from typing import Dict, Type
from .base import IngestionStrategy
from src.core.logging import logger

class PluginManager:
    """
    Dynamically loads and registers ingestion plugins.
    Allows easy extension without modifying core logic.
    """
    def __init__(self, plugin_package="src.ingestion.plugins"):
        self.plugin_package = plugin_package
        self.strategies: Dict[str, Type[IngestionStrategy]] = {}
        self._load_plugins()

    def _load_plugins(self):
        """
        Scans the plugin package directory, imports modules, and registers
        any class inheriting from IngestionStrategy.
        """
        # We need to find the directory path for the plugin package
        try:
            module = importlib.import_module(self.plugin_package)
            pkg_path = os.path.dirname(module.__file__)

            # Use pkgutil to iterate through modules in the package
            for _, module_name, is_pkg in pkgutil.iter_modules([pkg_path]):
                if not is_pkg:
                    full_module_name = f"{self.plugin_package}.{module_name}"
                    try:
                        # Import the module
                        module = importlib.import_module(full_module_name)

                        # Find classes in the module that subclass IngestionStrategy
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and
                                issubclass(attr, IngestionStrategy) and
                                attr is not IngestionStrategy):

                                # Register the strategy by its supported extensions
                                instance = attr()
                                for ext in instance.supported_extensions:
                                    self.strategies[ext.lower()] = attr
                                    logger.info(f"Registered plugin '{attr_name}' for extension '{ext}'")
                    except Exception as e:
                        logger.error(f"Failed to load plugin module {full_module_name}: {e}")
        except Exception as e:
            logger.error(f"Error during plugin discovery: {e}")

    def get_strategy(self, file_extension: str) -> IngestionStrategy:
        """
        Returns an instance of the IngestionStrategy registered for the given extension.

        Args:
            file_extension: The extension of the file (e.g., '.xlsx').

        Returns:
            An instance of the appropriate IngestionStrategy.

        Raises:
            ValueError: If no strategy is registered for the given extension.
        """
        ext = file_extension.lower()
        strategy_class = self.strategies.get(ext)
        if not strategy_class:
            raise ValueError(f"No ingestion strategy registered for extension: {ext}")
        return strategy_class()

# Create a global plugin manager instance
plugin_manager = PluginManager()
