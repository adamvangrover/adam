
import os
import yaml
import json
import re
from typing import Tuple, Dict, Any, Optional, Union, List
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Path to the prompt library relative to this file
PROMPT_LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../prompt_library'))

class PromptLoader:
    """
    Utility to load prompts from the filesystem.
    Supports simple text loading, JSON/YAML parsing, and 'Prompt-as-Code' style Markdown with YAML Frontmatter.
    """

    # Pre-compiled regex patterns for performance
    SYSTEM_PROMPT_PATTERN = re.compile(r"###\s*SYSTEM\s*PROMPT\s*\n(.*?)(?=###\s*USER\s*PROMPT|$)", re.DOTALL | re.IGNORECASE)
    USER_PROMPT_PATTERN = re.compile(r"###\s*USER\s*PROMPT\s*(?:###\s*TASK\s*PROMPT.*?)?\n(.*)", re.DOTALL | re.IGNORECASE)
    TASK_PROMPT_PATTERN = re.compile(r"###\s*TASK\s*PROMPT.*?\n(.*)", re.DOTALL | re.IGNORECASE)

    @staticmethod
    def get_path(prompt_name: str) -> str:
        """Finds the file path for a given prompt name (without extension)."""
        extensions = ['.md', '.json', '.yaml', '.yml', '.txt']

        # Check specific AOPL-v1.0 paths first for efficiency (Analyst OS)
        priority_paths = [
            os.path.join(PROMPT_LIB_PATH, "AOPL-v1.0/analyst_os", f"{prompt_name}.md"),
            os.path.join(PROMPT_LIB_PATH, "AOPL-v1.0", f"{prompt_name}.md"),
        ]

        for p in priority_paths:
            if os.path.exists(p):
                return p

        # Fallback to recursive search
        for root, dirs, files in os.walk(PROMPT_LIB_PATH):
            for ext in extensions:
                if f"{prompt_name}{ext}" in files:
                    return os.path.join(root, f"{prompt_name}{ext}")

        raise FileNotFoundError(f"Prompt '{prompt_name}' not found in {PROMPT_LIB_PATH}")

    @staticmethod
    def load_markdown_with_frontmatter(prompt_name: str) -> Tuple[Dict[str, Any], str, str]:
        """
        Parses a Markdown file with YAML frontmatter.
        Expected format:
        ---
        key: value
        ---
        ### SYSTEM PROMPT
        ...
        ### USER PROMPT
        ...

        Returns:
            (metadata, system_template, user_template)
        """
        path = PromptLoader.get_path(prompt_name)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split Frontmatter
        frontmatter = {}
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2]
                except yaml.YAMLError as e:
                    logger.error(f"Failed to parse YAML frontmatter for {prompt_name}: {e}")

        # Split System/User Prompts based on headers
        # We look for "### SYSTEM PROMPT" and "### USER PROMPT" (or similar variations)
        system_match = PromptLoader.SYSTEM_PROMPT_PATTERN.search(body)
        user_match = PromptLoader.USER_PROMPT_PATTERN.search(body)

        # Fallback regex if headers are slightly different or order is swapped
        if not user_match:
             # Try just finding the task prompt if user prompt header is missing
             user_match = PromptLoader.TASK_PROMPT_PATTERN.search(body)

        system_text = system_match.group(1).strip() if system_match else ""
        user_text = user_match.group(1).strip() if user_match else body.replace(system_match.group(0) if system_match else "", "").strip()

        return frontmatter, system_text, user_text

    @staticmethod
    def get(name: str) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Loads a prompt by name.
        If the file is JSON or YAML, parses and returns the data structure.
        Otherwise, returns the raw text content.
        """
        path = PromptLoader.get_path(name)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if path.endswith('.json'):
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON prompt {name}: {e}")
                return content

        if path.endswith(('.yaml', '.yml')):
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML prompt {name}: {e}")
                return content

        return content

# Legacy function support for backward compatibility
def load_prompt(prompt_name: str, type: str = "text") -> Any:
    """
    Deprecated: Use PromptLoader.get(prompt_name) instead.
    Preserves original signature for backward compatibility.
    """
    return PromptLoader.get(prompt_name)
