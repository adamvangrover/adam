# core/system/plugin_manager.py

import importlib
import os

from core.utils.config_utils import load_config


class PluginManager:
    def __init__(self, plugin_dir="plugins"):
        """
        Initializes the Plugin Manager.

        Args:
            plugin_dir (str): The directory where plugins are stored.
        """
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def load_plugins(self):
        """
        Loads plugins from the plugin directory.
        """
        for plugin_name in os.listdir(self.plugin_dir):
            if os.path.isdir(os.path.join(self.plugin_dir, plugin_name)):
                try:
                    import re
                    from pathlib import Path

                    if not re.match(r"^[a-zA-Z0-9_]+$", plugin_name):
                        print(f"Error loading plugin {plugin_name}: Invalid plugin name format")
                        continue

                    # Load plugin configuration
                    plugin_path = os.path.join(self.plugin_dir, plugin_name)
                    if not Path(plugin_path).resolve().is_relative_to(Path(self.plugin_dir).resolve()):
                        print(f"Error loading plugin {plugin_name}: Path traversal detected")
                        continue

                    config_path = os.path.join(plugin_path, "config.json")
                    plugin_config = load_config(config_path)

                    # Import plugin module
                    module_name = f"{self.plugin_dir}.{plugin_name}.{plugin_name}"
                    module = importlib.import_module(module_name)

                    # Instantiate plugin class safely
                    class_name = plugin_config.get("class_name")
                    if not class_name or not re.match(r"^[a-zA-Z0-9_]+$", class_name):
                        print(f"Error loading plugin {plugin_name}: Invalid class name format")
                        continue

                    plugin_class = getattr(module, class_name)
                    if not isinstance(plugin_class, type):
                        print(f"Error loading plugin {plugin_name}: Class {class_name} is not a valid type")
                        continue

                    plugin = plugin_class(plugin_config)

                    # Register plugin
                    self.plugins[plugin_name] = plugin

                except Exception as e:
                    print(f"Error loading plugin {plugin_name}: {e}")

    def get_plugin(self, plugin_name):
        """
        Retrieves a loaded plugin by name.

        Args:
            plugin_name (str): The name of the plugin.

        Returns:
            Plugin: The loaded plugin instance, or None if not found.
        """
        return self.plugins.get(plugin_name)

    def register_plugin(self, plugin_name, plugin):
        """
        Registers a plugin with the Plugin Manager.

        Args:
            plugin_name (str): The name of the plugin.
            plugin (Plugin): The plugin instance.
        """
        self.plugins[plugin_name] = plugin

    def unregister_plugin(self, plugin_name):
        """
        Unregisters a plugin from the Plugin Manager.

        Args:
            plugin_name (str): The name of the plugin.
        """
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
