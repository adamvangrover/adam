# core/utils/config_utils.py

import os
import re
import yaml
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

# Configure logger
logger = logging.getLogger(__name__)

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
        var_name, default_val = match.group(1), match.group(2)
        val = os.environ.get(var_name)

        if val is not None:
            return val
        if default_val is not None:
            return default_val

        logger.warning(f"Environment variable '{var_name}' not set and no default provided. Replacing with empty string.")
        return ""

    return ENV_VAR_PATTERN.sub(replace, content)

def load_config(file_path: str | Path) -> dict[str, Any] | None:
    """
    Loads a YAML configuration file with environment variable substitution.

    AI Context:
        Purpose: Parses a YAML file, substituting ${VAR} style env vars before parsing.
        Inputs: `file_path` (str or Path) pointing to the YAML file.
        Outputs: Returns a `dict` on success, or `None` on failure. Empty YAML returns `{}`.

    Args:
        file_path (str | Path): The path to the YAML configuration file.

    Returns:
        dict[str, Any] | None: The configuration as a dictionary, or None if an error occurred.
    """
    path = Path(file_path).resolve()

    if not path.is_file():
        logger.error(f"Config file not found: {path}")
        return None

    try:
        content = path.read_text(encoding='utf-8')
        content = _substitute_env_vars(content)

        config = yaml.safe_load(content)

        if config is None:
            logger.warning(f"Configuration file is empty: {path}")
            return {}
        if not isinstance(config, dict):
            logger.warning(f"Configuration file {path} did not parse as a dictionary. Returning empty dict.")
            return {}

        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in {path}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error loading config from {path}: {e}")
        return None

def deep_update(d: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively updates a dictionary. Fixes type errors when overriding scalar with dict.

    AI Context:
        Purpose: Merges two dictionaries recursively. The target dict `d` is updated with values from `u`.
        Inputs: `d` (base dict), `u` (update dict).
        Outputs: Returns the mutated dictionary `d`.

    Args:
        d (dict[str, Any]): The dictionary to update.
        u (dict[str, Any]): The dictionary containing new values.

    Returns:
        dict[str, Any]: The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            existing = d.get(k)
            if not isinstance(existing, dict):
                d[k] = {}
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

@lru_cache(maxsize=1)
def load_app_config() -> dict[str, Any]:
    """
    Loads and merges configurations from a predefined list of YAML files.

    Robustness Update:
    - Handles missing files gracefully (logs warning but continues).
    - Merges dictionaries deeply to prevent overwriting nested configurations.
    - Uses `@lru_cache` to drastically optimize performance by caching the combined dict.

    AI Context:
        Purpose: Aggregates configuration from all base YAML files into a single master dictionary.
        Outputs: Combined `dict`. Returns empty `{}` if none loaded. Cache can be cleared via `clear_config_cache()`.

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

    try:
        settings_idx = config_files_base.index("config/settings.yaml")
        config_files_ordered = (
            config_files_base[:settings_idx + 1] +
            ["config/agents.yaml"] +
            config_files_base[settings_idx + 1:]
        )
    except ValueError:
        logger.warning("config/settings.yaml not found in base config list, appending config/agents.yaml at the end.")
        config_files_ordered = config_files_base + ["config/agents.yaml"]

    combined_config: dict[str, Any] = {}
    files_loaded = 0

    for file_path in config_files_ordered:
        loaded = load_config(file_path)
        if loaded is not None:
            if isinstance(loaded, dict):
                deep_update(combined_config, loaded)
                files_loaded += 1
            else:
                logger.warning(f"Configuration file {file_path} content is not a dictionary. Skipping.")
        else:
            logger.warning(f"Configuration file {file_path} could not be loaded. Skipping.")

    if files_loaded == 0:
        logger.warning("No configuration files were successfully loaded. Application may not function correctly.")

    return combined_config

@lru_cache(maxsize=1)
def load_error_codes() -> dict[str, Any]:
    """
    Loads error codes from the errors.yaml configuration file. Uses caching for performance.

    Returns:
        dict[str, Any]: A dictionary containing error codes and messages.
    """
    error_config = load_config('config/errors.yaml')
    if error_config and isinstance(error_config, dict) and 'errors' in error_config:
        return error_config['errors']
    logger.warning("Could not load error codes from config/errors.yaml.")
    return {}

def clear_config_cache() -> None:
    """
    Clears the cached configurations for `load_app_config` and `load_error_codes`.
    Useful during testing or dynamic reconfiguration.
    """
    load_app_config.cache_clear()
    load_error_codes.cache_clear()

def save_config(config: dict[str, Any], file_path: str | Path) -> bool:
    """
    Saves a configuration dictionary to a YAML file.

    Args:
        config (dict[str, Any]): The configuration dictionary to save.
        file_path (str | Path): The path to the YAML file.

    Returns:
        bool: True if successful, False otherwise.
    """
    path = Path(file_path).resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {path}: {e}")
        return False
