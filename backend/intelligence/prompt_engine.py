import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging

class PromptEngine:
    """
    Manages loading, inheritance, and rendering of prompts.
    """

    def __init__(self, prompt_library_root: str = "prompt_library"):
        self.root = Path(prompt_library_root)
        self.logger = logging.getLogger(__name__)

    def load_prompt(self, relative_path: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Loads a prompt file, resolves inheritance, and renders variables.

        Args:
            relative_path: Path to the prompt file relative to prompt_library_root.
            context: Dictionary of variables to inject into the prompt.

        Returns:
            The rendered prompt string.

        Raises:
            FileNotFoundError: If the prompt file or its inherited parent cannot be found.
        """
        prompt_path = self.root / relative_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        raw_content = prompt_path.read_text(encoding="utf-8")

        # Handle inheritance
        # Check for '# INHERITS: <path>'
        # We assume it's on the second line or near the top
        lines = raw_content.splitlines()
        parent_content = ""

        for line in lines:
            if line.strip().startswith("# INHERITS:"):
                # Extract path
                parent_path_str = line.split(":", 1)[1].strip()
                # If path starts with prompt_library/, remove it to get relative path
                if parent_path_str.startswith("prompt_library/"):
                    parent_path_str = parent_path_str.replace("prompt_library/", "", 1)

                # Load parent content recursively (though usually just one level)
                # But here we just want the raw content of parent to prepend
                parent_full_path = self.root / parent_path_str
                if parent_full_path.exists():
                    parent_content = parent_full_path.read_text(encoding="utf-8")
                else:
                    # Critical safety check: Do not allow agents to run without their constitution
                    error_msg = f"Parent prompt not found: {parent_full_path} (inherited by {relative_path})"
                    self.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                break

        # Combine content
        full_content = ""
        if parent_content:
            full_content = parent_content + "\n\n" + raw_content
        else:
            full_content = raw_content

        # Render variables
        # Even if context is None, we might want to clean up placeholders
        if context is None:
            context = {}

        # Replace provided context variables
        for key, value in context.items():
            placeholder = "{" + key + "}"
            full_content = full_content.replace(placeholder, str(value))

        # Clean up standard keys if not provided
        standard_keys = ["input", "context", "memory", "tools", "code_context"]
        for key in standard_keys:
            placeholder = "{" + key + "}"
            if placeholder in full_content and (key not in context):
                full_content = full_content.replace(placeholder, "")

        return full_content
