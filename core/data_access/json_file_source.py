# core/data_access/json_file_source.py

import json
import logging
from pathlib import Path
from typing import Any

from core.data_access.base_data_source import BaseDataSource
from core.system.error_handler import FileReadError, InvalidInputError

logger = logging.getLogger(__name__)

class JsonFileSource(BaseDataSource):
    """
    Architecture & Usage:
    Data source for safely loading and parsing structured financial data from JSON files.
    This class inherits from BaseDataSource and implements strictly typed file I/O operations.

    Inputs are expected to be company IDs or generic data pointers.
    Outputs are dictionaries representing the deserialized structured data or None if unavailable.
    """

    def __init__(self, base_path: str = "data/") -> None:
        """Initialize the JsonFileSource with a target directory base path."""
        self.base_path = Path(base_path)

    def _load_json(self, file_path: Path) -> dict[str, Any] | None:
        """
        Safely loads and validates JSON data from a Path object.

        Args:
            file_path (Path): The pathlib Path to the target JSON file.

        Returns:
            dict[str, Any] | None: The parsed dictionary if successful, or None if invalid.

        Raises:
            FileReadError: If the file is missing or contains malformed JSON.
            InvalidInputError: If the top-level structure is not a dictionary.
        """
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                match data:
                    case dict():
                        return data
                    case _:
                        raise InvalidInputError(f"File {file_path} does not contain a valid JSON object at the top level.")
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            raise FileReadError(str(file_path), "File not found") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise FileReadError(str(file_path), f"Invalid JSON: {e}") from e
        except InvalidInputError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error loading data from {file_path}: {e}")
            return None

    def get_financial_statements(self, company_id: str) -> dict[str, Any] | None:
        """
        Retrieves financial statements for a specific company ID.

        Args:
            company_id (str): The unique identifier for the company.

        Returns:
            dict[str, Any] | None: Parsed financial statement data.
        """
        file_path = self.base_path / f"{company_id}_financials.json"
        return self._load_json(file_path)

    def get_historical_prices(self, company_id: str, start_date: str, end_date: str) -> dict[str, Any] | None:
        """
        Retrieves filtered historical prices for a company within a specific date range.

        Args:
            company_id (str): The unique identifier for the company.
            start_date (str): The ISO 8601 start date string (e.g., '2023-01-01').
            end_date (str): The ISO 8601 end date string (e.g., '2023-12-31').

        Returns:
            dict[str, Any] | None: A dictionary mapping dates to their respective prices.
        """
        file_path = self.base_path / f"{company_id}_prices.json"
        data = self._load_json(file_path)

        if data is None or not isinstance(data.get("prices"), list):
            logger.error(f"Invalid or missing 'prices' array in {file_path}")
            return None

        # Optimized filtering using dictionary comprehension
        return {
            item["date"]: item["price"]
            for item in data["prices"]
            if "date" in item and "price" in item and start_date <= item["date"] <= end_date
        }

    def get_market_data(self) -> dict[str, Any] | None:
        """
        Retrieves the global adam market baseline data.

        Returns:
            dict[str, Any] | None: Global market data payload.
        """
        file_path = self.base_path / "adam_market_baseline.json"
        return self._load_json(file_path)
