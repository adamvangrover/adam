# core/data_access/json_file_source.py
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from core.data_access.base_data_source import BaseDataSource
from core.system.error_handler import FileReadError, InvalidInputError

logger = logging.getLogger(__name__)


class JsonFileSource(BaseDataSource):
    """Data source for loading data from JSON files."""

    def __init__(self, base_path: str = "data/"):
        self.base_path = Path(base_path)

    def _load_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Loads and validates JSON data from a file."""
        try:
            with file_path.open('r') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise InvalidInputError(f"File {file_path} does not contain a valid JSON object at the top level.")
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise FileReadError(str(file_path), "File not found")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise FileReadError(str(file_path), f"Invalid JSON: {e}") from e
        except Exception as e:
            logging.exception(f"Error Loading Data:{e}")
            return None

    def get_financial_statements(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves financial statements for a company from a JSON file."""
        file_path = self.base_path / f"{company_id}_financials.json"  # Example file naming
        return self._load_json(file_path)

    def get_historical_prices(self, company_id: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieves historical prices for a company (simplified example)."""
        # In a real implementation, you'd likely have a more sophisticated way to handle time series data.
        file_path = self.base_path / f"{company_id}_prices.json"
        data = self._load_json(file_path)  # Load all price data.

        if data is None:  # Handle cases where data might be None
            return None

        if not isinstance(data, dict) or "prices" not in data or not isinstance(data["prices"], list):
            logger.error(f"Invalid data format in {file_path}")
            return None

        # Very basic filtering (replace with proper date handling).
        # This assumes a format like: {"prices": [{"date": "2023-01-01", "price": 150}, ... ]}
        filtered_prices = {}
        for item in data["prices"]:
            if item["date"] >= start_date and item["date"] <= end_date:
                filtered_prices[item["date"]] = item["price"]

        return filtered_prices

    def get_market_data(self) -> Optional[Dict[str, Any]]:
        """Retrieves general market data."""
        file_path = self.base_path / "adam_market_baseline.json"  # Example
        return self._load_json(file_path)
