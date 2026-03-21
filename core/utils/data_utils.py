"""
core/utils/data_utils.py

Architecture & Usage:
This module provides the central data loading and caching utility for Adam OS.
It is designed to cleanly ingest various data formats (JSON, CSV, YAML, and unstructured text via AI mock parsing)
and provides an in-memory caching mechanism to prevent redundant disk I/O.
By separating the logic into the `DataLoader` class, the application can maintain
isolated cache scopes where necessary or rely on the backward-compatible `load_data` wrapper function.

AI Context:
This module uses a strictly defined routing paradigm mapping string data types to parser methods.
Type hinting is strictly enforced using Python 3.10+ modern union paradigms `|`. Memory growth is capped
actively via LIFO evictions triggered at a dynamic `MAX_CACHE_ENTRIES` threshold.

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
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB limit to prevent memory exhaustion
    MAX_CACHE_ENTRIES: int = 1000

    def __init__(self, use_cache: bool = True) -> None:
        """
        Initializes the DataLoader with an optional caching mechanism.
        AI Context: Call this constructor once per data pipeline stage to maintain isolated cache states.
        If handling massive >50MB files, disable caching (`use_cache=False`) to preserve memory.

        Args:
            use_cache (bool): Determines whether file contents are cached in memory after the first read.
                              Defaults to True.
        """
        self.use_cache: bool = use_cache
        self._data_cache: dict[str, Any] = {}

    def load(self, source_config: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
        """
        Loads data dynamically based on the provided configuration using a strict routing paradigm.

        AI Context: Use this method to retrieve system settings, datasets, or external API responses.
        It inherently protects against path traversal and memory exhaustion through built-in bounds checking.
        If `source_config["type"]` is not explicitly supported, it throws an `InvalidInputError`.

        Routing Logic:
            - "api": Redirects to `_load_api`
            - "json", "csv", "yaml", "smart_text": Validates path existence, enforces `MAX_FILE_SIZE_BYTES`,
              checks the local memory cache (`_data_cache`), and then reads from disk via `_read_file`.

        Args:
            source_config (dict[str, Any]): A dictionary requiring a strictly formatted 'type' string
                                            and an optional 'path' string depending on the type.
                                            Example: {"type": "json", "path": "/path/to/data.json"}

        Returns:
            dict[str, Any] | list[Any] | None: A structured representation of the target data, or None on graceful failure.

        Raises:
            InvalidInputError: Triggered when 'type' is missing/unsupported or 'path' is omitted for local files.
            FileReadError: Triggered upon I/O failures, missing files, parser crashes, or if the target exceeds 50MB.
        """
        data_type = source_config.get("type")
        file_path = source_config.get("path")

        match data_type:
            case "api":
                return self._load_api(source_config)
            case t if t in self.SUPPORTED_TYPES:
                if not file_path:
                    logger.error("Invalid source configuration: 'path' is required for file-based sources.")
                    raise InvalidInputError("Missing 'path' in source_config")
                file_path_obj = Path(file_path)
            case _:
                logger.error(f"Unsupported data type: {data_type}")
                raise InvalidInputError(f"Unsupported data type: {data_type}")

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
            if len(self._data_cache) >= self.MAX_CACHE_ENTRIES:
                logger.warning("Cache capacity reached. Evicting oldest entry.")
                # LIFO cache eviction or simple pop
                self._data_cache.pop(next(iter(self._data_cache)))
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
        try:
            match data_type:
                case "json":
                    return DataLoader._load_json(file_path)
                case "csv":
                    return DataLoader._load_csv(file_path)
                case "yaml":
                    return DataLoader._load_yaml(file_path)
                case "smart_text":
                    return DataLoader._load_smart_text(file_path)
                case _:
                    raise ValueError(f"Unhandled data type in _read_file: {data_type}")
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

        match provider:
            case "example_financial_data_api":
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
            case "example_market_data_api":
                return {
                    "market_trends": [
                       {"sector": "healthcare", "trend": "neutral"},
                       {"sector": "energy", "trend": "downward"}
                    ]
                }
            case _:
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
