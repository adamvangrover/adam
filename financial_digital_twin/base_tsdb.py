from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List


class BaseTSDBClient(ABC):
    """
    Abstract base class for a time-series database client.
    """

    @abstractmethod
    def connect(self):
        """
        Connect to the time-series database.
        """
        pass

    @abstractmethod
    def query(self, entity_id: str, metric: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Query time-series data for a given entity and metric.

        :param entity_id: The ID of the entity to query for.
        :param metric: The name of the metric to query.
        :param start_date: The start date of the query range.
        :param end_date: The end date of the query range.
        :return: A list of data points, where each data point is a dictionary.
        """
        pass

    @abstractmethod
    def write(self, entity_id: str, metric: str, data: List[Dict[str, Any]]):
        """
        Write time-series data for a given entity and metric.

        :param entity_id: The ID of the entity to write for.
        :param metric: The name of the metric to write.
        :param data: A list of data points to write.
        """
        pass
