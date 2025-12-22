
import os
import yaml
import json
from typing import Any, Dict, Optional

from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Path to the prompt library relative to this file
# core/prompting/loader.py -> ../../prompt_library
PROMPT_LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../prompt_library'))


def load_prompt(prompt_name: str, type: str = "text") -> Any:
    """
    Loads a prompt from the library by name.

    Args:
        prompt_name (str): The name of the prompt (filename without extension).
        type (str): 'text' (returns content) or 'json'/'yaml' (returns dict).

    Returns:
        The prompt content or data structure.
    """
    logger.info(f"Loading prompt: {prompt_name}")

    # Search for files with supported extensions
    extensions = ['.md', '.json', '.yaml', '.yml', '.txt']

    found_path = None
    for ext in extensions:
        # Check root of prompt_library
        path = os.path.join(PROMPT_LIB_PATH, f"{prompt_name}{ext}")
        if os.path.exists(path):
            found_path = path
            break

        # Check immediate subdirectories (e.g. AOPL-v1.0)
        # For simplicity, we can just walk the directory

    if not found_path:
        # Deep search
        for root, dirs, files in os.walk(PROMPT_LIB_PATH):
            for ext in extensions:
                if f"{prompt_name}{ext}" in files:
                    found_path = os.path.join(root, f"{prompt_name}{ext}")
                    break
            if found_path:
                break

    if not found_path:
        raise FileNotFoundError(f"Prompt '{prompt_name}' not found in {PROMPT_LIB_PATH}")

    with open(found_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if found_path.endswith('.json'):
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON prompt {prompt_name}: {e}")
            return content

    if found_path.endswith(('.yaml', '.yml')):
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML prompt {prompt_name}: {e}")
            return content

    return content


class PromptLoader:
    """Class wrapper for convenience."""
    @staticmethod
    def get(name: str) -> Any:
        return load_prompt(name)
