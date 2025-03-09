# core/system/knowledge_base.py

import json
import logging
from typing import Optional
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeBase:
    """
    A simple knowledge base that loads data from a JSON file.
    """

    def __init__(self, file_path: str = "data/knowledge_base.json"):
        """
        Initializes the KnowledgeBase.

        Args:
            file_path: The path to the JSON file containing the knowledge base data.
        """
        self.file_path = Path(file_path)
        self.data = self._load_data()

    def _load_data(self) -> Optional[dict]:
        """Loads data from the JSON file."""
        if not self.file_path.exists():
            logging.error(f"Knowledge base file not found: {self.file_path}")
            return None  # Return None if the file doesn't exist

        try:
            with self.file_path.open('r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading knowledge base from {self.file_path}: {e}")
            return None  # Return None on error

    def query(self, query_key: str) -> Optional[str]:
        """
        Queries the knowledge base. This is a *very* simple key-based lookup.
        A real implementation would likely have more sophisticated search capabilities.

        Args:
            query_key: The key to look up in the knowledge base.

        Returns:
            The value associated with the key, or None if not found.
        """
        if self.data:
            return self.data.get(query_key)
        else:
            logging.warning("Knowledge base data is empty.")
            return None
