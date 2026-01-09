# core/utils/config_utils.py

import yaml
import os  # Import the 'os' module
import logging
from typing import Dict, Any, Optional


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
            if config is None:  # Handle empty YAML files
                logging.warning(f"Configuration file is empty: {file_path}")
                return {}  # Return empty dict
            return config
    except FileNotFoundError:
        # Logging as error might be too noisy for optional configs, using warning
        # But tests expect ERROR for non-existent files in direct load_config calls
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

    Robustness Update:
    - Handles missing files gracefully (logs warning but continues).
    - Merges dictionaries deeply if needed (currently shallow update).
    - Returns empty dict if no configs loaded instead of crashing.

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
        "config/settings.yaml",  # agents.yaml will be added after this
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
        config_files_ordered = config_files_base[:settings_index+1] + \
            ["config/agents.yaml"] + config_files_base[settings_index+1:]
    except ValueError:
        # Fallback if settings.yaml is not in the list for some reason, just append agents.yaml
        logging.warning("config/settings.yaml not found in base config list, appending config/agents.yaml at the end.")
        config_files_ordered = config_files_base + ["config/agents.yaml"]

    combined_config: Dict[str, Any] = {}
    files_loaded = 0

    for file_path in config_files_ordered:
        # We handle FileNotFoundError inside load_config, which logs ERROR.
        # But for app startup, missing some config files might be acceptable (WARN)
        # or critical (ERROR). Since load_config already logs ERROR on missing,
        # we can just skip here.
        loaded_content = load_config(file_path)
        if loaded_content is not None:
            if isinstance(loaded_content, dict):
                combined_config.update(loaded_content)
                files_loaded += 1
            else:
                logging.warning(f"Configuration file {file_path} content is not a dictionary. Skipping.")
        else:
            # Already logged in load_config
            # Ideally we would downgrade log level for optional files, but
            # load_config is generic.
            logging.warning(f"Configuration file {file_path} could not be loaded. Skipping.")
            pass

    if files_loaded == 0:
        logging.warning("No configuration files were successfully loaded. Application may not function correctly.")

    return combined_config


def load_error_codes() -> Dict[str, Any]:
    """
    Loads error codes from the errors.yaml configuration file.

    Returns:
        A dictionary containing error codes and messages.
    """
    error_config = load_config('config/errors.yaml')
    if error_config and 'errors' in error_config:
        return error_config['errors']
    logging.warning("Could not load error codes from config/errors.yaml.")
    return {}


def save_config(config: Dict[str, Any], file_path: str) -> bool:
    """
    Saves a configuration dictionary to a YAML file.

    Args:
        config: The configuration dictionary to save.
        file_path: The path to the YAML file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        logging.error(f"Error saving config to {file_path}: {e}")
        return False
