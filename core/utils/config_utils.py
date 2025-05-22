# core/utils/config_utils.py

import yaml
import os # Import the 'os' module
import logging
from typing import Dict, Any, Optional
#from core.system.error_handler import ConfigurationError # Removed as per instructions to simplify error handling


# Configure logging (consider moving to a central location)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads a YAML configuration file.

    Args:
        file_path: The path to the YAML configuration file.

    Returns:
        The configuration as a dictionary, or None if an error occurred.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None: # Handle empty YAML files
                logging.warning(f"Configuration file is empty: {file_path}")
                return {} # Return empty dict
            return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML in {file_path}: {e}")
        return None
    except Exception as e:
        logging.exception(f"Error loading config from {file_path}: {e}")
        return None

def load_app_config() -> Dict[str, Any]:
    """
    Loads and merges configurations from a predefined list of YAML files.

    Returns:
        A dictionary containing the combined configuration.
    """
    config_files_base = [
        "config/api.yaml",
        "config/knowledge_graph.yaml",
        "config/data_sources.yaml",
        "config/analysis_modules.yaml",
        "config/reporting.yaml",
        "config/logging.yaml",
        "config/system.yaml",
        "config/settings.yaml", # agents.yaml will be added after this
        "config/workflow.yaml",
        "config/api_keys.yaml",
        "config/cacm-adk-config.yaml",
        "config/errors.yaml",
        "config/knowledge_graph_schema.yaml",
        "config/llm_plugin.yaml",
        "config/newsletter_layout.yaml",
        "config/report_layout.yaml",
    ]

    # Ensure agents.yaml is loaded after settings.yaml
    # Find index of settings.yaml and insert agents.yaml after it
    try:
        settings_index = config_files_base.index("config/settings.yaml")
        config_files_ordered = config_files_base[:settings_index+1] + ["config/agents.yaml"] + config_files_base[settings_index+1:]
    except ValueError:
        # Fallback if settings.yaml is not in the list for some reason, just append agents.yaml
        logging.warning("config/settings.yaml not found in base config list, appending config/agents.yaml at the end.")
        config_files_ordered = config_files_base + ["config/agents.yaml"]


    combined_config: Dict[str, Any] = {}
    for file_path in config_files_ordered:
        loaded_content = load_config(file_path)
        if loaded_content is not None:
            # Simple update is fine as each file should have its own top-level key mostly
            # For agents.yaml vs settings.yaml, if both have an 'agents' key,
            # loading agents.yaml after settings.yaml ensures its 'agents' key prevails.
            combined_config.update(loaded_content)
        else:
            logging.warning(f"Configuration file {file_path} could not be loaded. Skipping.")
    
    return combined_config

# Example Usage (you can remove this when you integrate it into your project)
if __name__ == '__main__':
    # Create a dummy config file for testing
    dummy_config = {"key1": "value1", "key2": 123}
    with open("test_config.yaml", "w") as f:
        yaml.dump(dummy_config, f)

    # Load the dummy config
    config = load_config("test_config.yaml")
    print(f"Loaded config: {config}")

    # Test with a non-existent file
    config = load_config("nonexistent_file.yaml")
    print(f"Loaded config (nonexistent): {config}")

    # Test with an invalid YAML file.
    with open("invalid.yaml", "w") as f:
        f.write("key1: value1\nkey2: - item1\n  -item2")  # Invalid YAML
    config = load_config("invalid.yaml")
    print(f"Loaded config (invalid YAML): {config}")

    # Test with an empty yaml file
    with open("empty.yaml", "w") as f:
        pass # Creates an empty file
    config = load_config("empty.yaml")
    print(f"Loaded config (empty YAML): {config}")

    # Cleanup the test files
    os.remove("test_config.yaml")
    os.remove("invalid.yaml")
    os.remove("empty.yaml")
