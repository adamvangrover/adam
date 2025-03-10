# core/data_access/base_data_source.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.
    Defines the common interface for accessing data.
    """

    @abstractmethod
    def get_financial_statements(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves financial statements for a company."""
        pass

    @abstractmethod
    def get_historical_prices(self, company_id: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieves historical prices for a company."""
        pass

    # Add other common data retrieval methods as needed

    @abstractmethod
    def get_market_data(self) -> Optional[Dict[str, Any]]:
      """Placeholder for retrieving general market data."""
      pass
