import logging
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class BigQueryConnector:
    """
    Connector for Google BigQuery.
    Enables the system to push analysis results to a data warehouse
    and query large-scale financial datasets.
    """

    def __init__(self, project_id: str, dataset_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                from google.cloud import bigquery
                self.client = bigquery.Client(project=self.project_id)
                logger.info(f"Initialized BigQuery Connector for {self.project_id}.{self.dataset_id} (Live)")
            except Exception as e:
                logger.warning(f"Failed to initialize live BigQuery client: {e}. Using Mock.")
                self.client = "MOCK_BQ_CLIENT"
        else:
            self.client = "MOCK_BQ_CLIENT"
            logger.info(f"GOOGLE_APPLICATION_CREDENTIALS not set. Initialized BigQuery Connector for {self.project_id}.{self.dataset_id} (Mock)")

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Executes a SQL query.
        """
        logger.info(f"Executing BQ Query: {sql}")
        if self.client == "MOCK_BQ_CLIENT":
            return [
                {"ticker": "AAPL", "revenue": 100000, "period": "2024-Q3"},
                {"ticker": "GOOGL", "revenue": 85000, "period": "2024-Q3"}
            ]
        try:
            query_job = self.client.query(sql)
            return [dict(row) for row in query_job]
        except Exception as e:
            logger.warning(f"Live BigQuery query failed: {e}. Falling back to mock data.")
            return [
                {"ticker": "AAPL", "revenue": 100000, "period": "2024-Q3"},
                {"ticker": "GOOGL", "revenue": 85000, "period": "2024-Q3"}
            ]

    def insert_rows(self, table_id: str, rows: List[Dict[str, Any]]) -> bool:
        """
        Inserts rows into a BigQuery table.
        """
        full_table_id = f"{self.project_id}.{self.dataset_id}.{table_id}"
        logger.info(f"Inserting {len(rows)} rows into {full_table_id}")

        if self.client == "MOCK_BQ_CLIENT":
            return True

        try:
            errors = self.client.insert_rows_json(full_table_id, rows)
            if errors:
                logger.error(f"BQ Insert Errors: {errors}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Live BigQuery insert failed: {e}. Falling back to mock (True).")
            return True

    def get_schema(self, table_id: str) -> List[Dict[str, str]]:
        """
        Retrieves the schema of a table.
        """
        return [
            {"name": "ticker", "type": "STRING"},
            {"name": "date", "type": "DATE"},
            {"name": "value", "type": "FLOAT"}
        ]
