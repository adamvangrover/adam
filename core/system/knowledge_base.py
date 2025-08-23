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

    def _load_data(self) -> dict:
        """Loads data from the JSON file and converts keys to lowercase."""
        if not self.file_path.exists():
            logging.error(f"Knowledge base file not found: {self.file_path}")
            return {}  # Return empty dict if the file doesn't exist

        try:
            with self.file_path.open('r') as f:
                data = json.load(f)
                return {k.lower(): v for k, v in data.items()}
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading knowledge base from {self.file_path}: {e}")
            return {}  # Return empty dict on error

    def query(self, query_key: str) -> Optional[str]:
        """
        Queries the knowledge base. This is a *very* simple key-based lookup.
        A real implementation would likely have more sophisticated search capabilities.
        Keys are queried in lowercase.

        Args:
            query_key: The key to look up in the knowledge base.

        Returns:
            The value associated with the key, or None if not found.
        """
        if self.data:
            return self.data.get(query_key.lower())
        else:
            logging.warning("Knowledge base data is empty.")
            return None

    def update(self, key: str, value: any):
        """
        Updates the knowledge base with a new key-value pair.
        Keys are stored in lowercase.
        This updates the in-memory dictionary only.
        Call save() to persist changes to the file.
        """
        if self.data is None:
            self.data = {}
        self.data[key.lower()] = value

    def save(self):
        """Saves the current knowledge base data to the JSON file."""
        if self.data is None:
            logging.warning("No data to save.")
            return

        try:
            with self.file_path.open('w') as f:
                json.dump(self.data, f, indent=4)
            logging.info(f"Knowledge base saved to {self.file_path}")
        except IOError as e:
            logging.error(f"Error saving knowledge base to {self.file_path}: {e}")
