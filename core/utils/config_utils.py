# core/utils/config_utils.py

import yaml
import os
import logging
import re
from pathlib import Path
from typing import Any

# Configure logging (consider moving to a central location)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🛡️ Sentinel: Regex for finding environment variables in the format ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r'\$\{(\w+)(?::([^}]+))?\}')

def _substitute_env_vars(content: str) -> str:
    """
    🛡️ Sentinel: Substitute environment variables in the format ${VAR} or ${VAR:default} within a string.
    If the variable is not set and no default is provided, it is replaced with an empty string.

    Args:
        content (str): The raw string content containing potential environment variables.

    Returns:
        str: The content with environment variables substituted.
    """
    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        default_val = match.group(2)
        val = os.environ.get(var_name)

        if val is not None:
            return val
        if default_val is not None:
            return default_val

        logging.warning(f"Environment variable '{var_name}' not set and no default provided. Replacing with empty string.")
        return ""

    return ENV_VAR_PATTERN.sub(replace, content)

def load_config(file_path: str | Path) -> dict[str, Any] | None:
    """
    Loads a YAML configuration file with environment variable substitution.

    Args:
        file_path (str | Path): The path to the YAML configuration file.

    Returns:
        dict[str, Any] | None: The configuration as a dictionary, or None if an error occurred.
    """
    path = Path(file_path)

    if not path.exists():
        logging.error(f"Config file not found: {path}")
        return None

    try:
        content = path.read_text(encoding='utf-8')

        # 🛡️ Sentinel: Substitute environment variables before parsing
        content = _substitute_env_vars(content)

        config = yaml.safe_load(content)
        if config is None:  # Handle empty YAML files
            logging.warning(f"Configuration file is empty: {path}")
            return {}
        if not isinstance(config, dict):
            logging.warning(f"Configuration file {path} did not parse as a dictionary. Returning empty dict.")
            return {}

        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML in {path}: {e}")
        return None
    except Exception as e:
        logging.exception(f"Error loading config from {path}: {e}")
        return None

def deep_update(d: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively updates a dictionary. Fixes type errors when overriding scalar with dict.

    Args:
        d (dict[str, Any]): The dictionary to update.
        u (dict[str, Any]): The dictionary containing new values.

    Returns:
        dict[str, Any]: The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            existing = d.get(k, {})
            if not isinstance(existing, dict):
                existing = {}
            d[k] = deep_update(existing, v)
        else:
            d[k] = v
    return d

def load_app_config() -> dict[str, Any]:
    """
    Loads and merges configurations from a predefined list of YAML files.

    Robustness Update:
    - Handles missing files gracefully (logs warning but continues).
    - Merges dictionaries deeply to prevent overwriting nested configurations.
    - Returns empty dict if no configs loaded instead of crashing.

    Returns:
        dict[str, Any]: A dictionary containing the combined configuration.
    """
    config_files_base = [
        "config/api.yaml",
        "config/knowledge_graph.yaml",
        "config/data_sources.yaml",
        "config/analysis_modules.yaml",
        "config/reporting.yaml",
        "config/logging.yaml",
        "config/system.yaml",
        "config/settings.yaml",
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
    try:
        settings_index = config_files_base.index("config/settings.yaml")
        config_files_ordered = (
            config_files_base[:settings_index + 1] +
            ["config/agents.yaml"] +
            config_files_base[settings_index + 1:]
        )
    except ValueError:
        logging.warning("config/settings.yaml not found in base config list, appending config/agents.yaml at the end.")
        config_files_ordered = config_files_base + ["config/agents.yaml"]

    combined_config: dict[str, Any] = {}
    files_loaded = 0

    for file_path in config_files_ordered:
        loaded_content = load_config(file_path)
        if loaded_content is not None:
            if isinstance(loaded_content, dict):
                combined_config = deep_update(combined_config, loaded_content)
                files_loaded += 1
            else:
                logging.warning(f"Configuration file {file_path} content is not a dictionary. Skipping.")
        else:
            logging.warning(f"Configuration file {file_path} could not be loaded. Skipping.")

    if files_loaded == 0:
        logging.warning("No configuration files were successfully loaded. Application may not function correctly.")

    return combined_config

def load_error_codes() -> dict[str, Any]:
    """
    Loads error codes from the errors.yaml configuration file.

    Returns:
        dict[str, Any]: A dictionary containing error codes and messages.
    """
    error_config = load_config('config/errors.yaml')
    if error_config and isinstance(error_config, dict) and 'errors' in error_config:
        return error_config['errors']
    logging.warning("Could not load error codes from config/errors.yaml.")
    return {}

def save_config(config: dict[str, Any], file_path: str | Path) -> bool:
    """
    Saves a configuration dictionary to a YAML file.

    Args:
        config (dict[str, Any]): The configuration dictionary to save.
        file_path (str | Path): The path to the YAML file.

    Returns:
        bool: True if successful, False otherwise.
    """
    path = Path(file_path)
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        logging.error(f"Error saving config to {path}: {e}")
        return False
