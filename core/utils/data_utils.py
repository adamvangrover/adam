"""
core/utils/data_utils.py

Architecture & Usage:
This module provides the central data loading and caching utility for Adam OS.
It is designed to cleanly ingest various data formats (JSON, CSV, YAML, and unstructured text via AI mock parsing)
and provides an in-memory caching mechanism to prevent redundant disk I/O.
By separating the logic into the `DataLoader` class, the application can maintain
isolated cache scopes where necessary or rely on the backward-compatible `load_data` wrapper function.

Typical Usage Example:
    loader = DataLoader(use_cache=True)
    data = loader.load({"type": "json", "path": "config/settings.json"})

Dependencies:
    - Built-ins: csv, json, pathlib, typing
    - External: yaml
"""


import csv
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from core.system.error_handler import FileReadError, InvalidInputError

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to handle loading data from files (JSON, CSV, YAML) or APIs,
    with built-in caching.
    """
    SUPPORTED_TYPES = {"json", "csv", "yaml", "smart_text"}
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit to prevent memory exhaustion

    def __init__(self, use_cache: bool = True):
        """
        Initializes the DataLoader with an optional caching mechanism.

        Args:
            use_cache (bool): Determines whether file contents are cached in memory after the first read.
                              Defaults to True.
        """
        self.use_cache = use_cache
        self._data_cache = {}

    def load(self, source_config: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
        """
        Loads data based on the provided source configuration.

        This method identifies the data type, validates file presence and size bounds,
        utilizes the cache if requested, and falls back to actual file reading/API querying.

        Args:
            source_config (dict[str, Any]): A dictionary containing 'type' and 'path' keys.
                                            Example: {"type": "json", "path": "/path/to/file.json"}

        Returns:
            dict[str, Any] | list[Any] | None: The loaded data structure, or None if load fails.

        Raises:
            InvalidInputError: If 'type' or 'path' are missing or invalid.
            FileReadError: If the file does not exist, exceeds the size limit, or fails to parse.
        """
        data_type = source_config.get("type")
        file_path = source_config.get("path")

        if data_type == "api":
            return self._load_api(source_config)

        if not file_path:
            logger.error("Invalid source configuration: 'path' is required for file-based sources.")
            raise InvalidInputError("Missing 'path' in source_config")

        if data_type not in self.SUPPORTED_TYPES:
            logger.error(f"Unsupported data type: {data_type}")
            raise InvalidInputError(f"Unsupported data type: {data_type}")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error(f"Data file not found: {file_path_obj}")
            raise FileReadError(str(file_path_obj), "File not found")

        file_size = file_path_obj.stat().st_size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            logger.error(f"File {file_path_obj} exceeds maximum allowed size ({self.MAX_FILE_SIZE_BYTES} bytes)")
            raise FileReadError(str(file_path_obj), f"File exceeds maximum size limit of {self.MAX_FILE_SIZE_BYTES} bytes")

        cache_key = str(file_path_obj)
        if self.use_cache and cache_key in self._data_cache:
            logger.info(f"Loading data from cache: {file_path_obj}")
            return self._data_cache[cache_key]

        data = self._read_file(file_path_obj, data_type)
        if self.use_cache and data is not None:
            self._data_cache[cache_key] = data
        return data

    @staticmethod
    def _read_file(file_path: Path, data_type: str) -> dict[str, Any] | list[Any] | None:
        """
        Internal method to handle the routing and parsing of specific file types.

        Args:
            file_path (Path): Pathlib object pointing to the target file.
            data_type (str): The string identifier for the file type ('json', 'csv', 'yaml', 'smart_text').

        Returns:
            The parsed data structure depending on the file type.
        """
        parsers = {
            "json": DataLoader._load_json,
            "csv": DataLoader._load_csv,
            "yaml": DataLoader._load_yaml,
            "smart_text": DataLoader._load_smart_text,
        }

        parser = parsers.get(data_type)
        try:
            if not parser:
                raise ValueError(f"Unhandled data type in _read_file: {data_type}")
            return parser(file_path)
        except (json.JSONDecodeError, FileNotFoundError, IOError, yaml.YAMLError) as e:
            logger.exception(f"Error loading data from {file_path}: {e}")
            raise FileReadError(str(file_path), str(e)) from e
        except Exception as e:
            logger.exception(f"Unexpected error loading data from {file_path}: {e}")
            raise FileReadError(str(file_path), str(e)) from e

    @staticmethod
    def _load_json(file_path: Path) -> dict[str, Any]:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _load_csv(file_path: Path) -> list[dict[str, Any]]:
        with file_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def _load_yaml(file_path: Path) -> dict[str, Any]:
        with file_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_smart_text(file_path: Path) -> dict[str, Any]:
        """
        AI Feature: Parses unstructured text into a structured JSON representation
        using a mock LLM extraction pipeline. In production, this would call
        an actual LLM API or local model to extract named entities and relationships.
        """
        logger.info(f"Initiating AI smart parsing for unstructured text: {file_path}")
        with file_path.open('r', encoding="utf-8") as f:
            raw_text = f.read()

        # Mock LLM API extraction logic
        # Here we simulate an advanced RAG feature extracting complex insights
        mock_extracted_data = {
            "source_file": str(file_path),
            "text_length": len(raw_text),
            "extracted_entities": ["MockCompany Inc.", "CEO John Doe", "Q3 Revenue $50M"],
            "summary": "This is a mock AI-generated summary of the unstructured text.",
            "rag_context": {
                "vector_db_matches": 3,
                "confidence_score": 0.92,
                "keywords": ["finance", "revenue", "growth"]
            }
        }
        return mock_extracted_data

    def _load_api(self, source_config: dict[str, Any]) -> dict[str, Any] | None:
        """Provides placeholder data for API calls."""
        logger.warning("API data source not yet implemented. Returning placeholder data.")
        provider = source_config.get("provider")

        if provider == "example_financial_data_api":
            return {
                "income_statement": {
                    "revenue": [1000, 1100, 1250],
                    "net_income": [100, 120, 150],
                },
                "balance_sheet": {
                    "assets": [2000, 2100, 2200],
                    "liabilities": [800, 850, 900],
                },
                "cash_flow_statement": {
                    "operating_cash_flow": [150, 170, 200]
                }
            }
        elif provider == "example_market_data_api":
            return {
                "market_trends": [
                   {"sector": "healthcare", "trend": "neutral"},
                   {"sector": "energy", "trend": "downward"}
                ]
            }
        else:
            logger.warning(f"Unknown API provider: {provider}")
            return None


def load_data(source_config: dict[str, Any], cache: bool = True) -> dict[str, Any] | list[Any] | None:
    """
    Wrapper for DataLoader to maintain backward compatibility.
    Loads data from a file (JSON or CSV) or a placeholder for an API, based on configuration.

    Args:
        source_config (dict[str, Any]): The configuration dictionary specifying 'type' and 'path'.
        cache (bool): Whether to use in-memory caching. Defaults to True.

    Returns:
        dict[str, Any] | list[Any] | None: The loaded data or None on failure.
    """
    loader = DataLoader(use_cache=cache)
    return loader.load(source_config)
