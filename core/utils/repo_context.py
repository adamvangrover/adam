import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RepoContextManager:
    """
    Singleton to load and parse repository documentation (AGENTS.md, README.md)
    to provide dynamic context for Agents and Routers.
    """
    _instance = None
    _context_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RepoContextManager, cls).__new__(cls)
            cls._instance._load_context()
        return cls._instance

    def _load_context(self):
        """Loads key documentation files."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.files = {
            "AGENTS.md": os.path.join(root_dir, "AGENTS.md"),
            "README.md": os.path.join(root_dir, "README.md")
        }

        for name, path in self.files.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._context_cache[name] = f.read()
                    logger.info(f"Loaded {name} into Repo Context.")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
                    self._context_cache[name] = ""
            else:
                logger.warning(f"{name} not found at {path}")
                self._context_cache[name] = ""

    def get_context(self, file_key: str) -> str:
        """Returns the content of a specific file."""
        return self._context_cache.get(file_key, "")

    def get_agent_definitions(self) -> str:
        """
        Extracts the Agent Architecture section from AGENTS.md.
        Useful for initializing the MetaOrchestrator's mental model.
        """
        content = self.get_context("AGENTS.md")
        if not content:
            return "No Agent Definitions Available."

        # Simple slicing for now, can be made more robust with Regex
        start_marker = "## Agent Architecture"
        end_marker = "## Getting Started"

        try:
            start_idx = content.index(start_marker)
            end_idx = content.index(end_marker)
            return content[start_idx:end_idx]
        except ValueError:
            return content # Return full file if markers missing

    def get_directives(self) -> str:
        """Returns coding standards and directives."""
        content = self.get_context("AGENTS.md")
        start_marker = "## Directives for v25 Development"
        try:
            start_idx = content.index(start_marker)
            return content[start_idx:]
        except ValueError:
            return ""
