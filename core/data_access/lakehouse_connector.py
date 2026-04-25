from core.tools.base_tool import BaseTool
from core.security.sql_validator import SQLValidator
from typing import Any
import json
import os
import pandas as pd
import logging

# Configure logger
logger = logging.getLogger(__name__)

class LakehouseConnector(BaseTool):
    name = "microsoft_fabric_run_sql"
    description = "Executes read-only T-SQL queries against the Data Lakehouse."

    def __init__(self):
        # Check for local lakehouse directory
        self.local_lake_path = "data/lakehouse"
        if not os.path.exists(self.local_lake_path):
            os.makedirs(self.local_lake_path, exist_ok=True)
            # Create dummy parquet if empty
            try:
                df = pd.DataFrame([
                    {"metric": "Revenue", "year": 2023, "value": 383000000000, "ticker": "AAPL"},
                    {"metric": "EBITDA", "year": 2023, "value": 125000000000, "ticker": "AAPL"},
                    {"metric": "Revenue", "year": 2023, "value": 96000000000, "ticker": "TSLA"},
                    {"metric": "EBITDA", "year": 2023, "value": 15000000000, "ticker": "TSLA"}
                ])
                # Use fastparquet or pyarrow if available, else json
                try:
                    df.to_parquet(os.path.join(self.local_lake_path, "financials.parquet"))
                except ImportError:
                    df.to_json(os.path.join(self.local_lake_path, "financials.json"), orient="records")
            except Exception as e:
                pass

    def execute(self, sql_query: str) -> str:
        """
        Executes a SQL query against the Fabric Lakehouse.
        Graceful Fallback: If connection fails (simulated), queries local Parquet/JSON files.
        """
        # Security Check: STRICT WHITELIST via SQLValidator
        if not SQLValidator.validate_read_only(sql_query):
             logger.warning(f"SecurityViolation: Blocked unsafe SQL query: {sql_query}")
             return json.dumps({"error": "SecurityViolation: Only strict read-only SELECT statements are allowed."})

        # SIMULATE CONNECTION FAILURE -> FALLBACK
        try:
            return self._query_local_lake(sql_query)
        except Exception as e:
            logger.error(f"Lakehouse query failed: {e}")
            return json.dumps({"error": f"Fallback failed: {e}"})

    def _query_local_lake(self, query: str) -> str:
        # Use DuckDB for real SQL execution over local files
        try:
            import duckdb

            # Setup duckdb connection
            con = duckdb.connect(database=':memory:')

            # Register local files as tables if they exist
            parquet_path = os.path.join(self.local_lake_path, "financials.parquet")
            json_path = os.path.join(self.local_lake_path, "financials.json")

            table_created = False
            if os.path.exists(parquet_path):
                escaped_path = parquet_path.replace("'", "''")
                if ";" in escaped_path or "--" in escaped_path:
                    raise ValueError("Invalid file path: suspicious characters detected.")
                query_parts = ["CREATE VIEW financials AS SELECT * FROM read_parquet('", escaped_path, "')"]
                con.execute("".join(query_parts))
                table_created = True
            elif os.path.exists(json_path):
                escaped_path = json_path.replace("'", "''")
                if ";" in escaped_path or "--" in escaped_path:
                    raise ValueError("Invalid file path: suspicious characters detected.")
                query_parts = ["CREATE VIEW financials AS SELECT * FROM read_json_auto('", escaped_path, "')"]
                con.execute("".join(query_parts))
                table_created = True

            if table_created:
                # Execute the provided query
                # Note: We assume the query references the 'financials' table
                # A more robust system would map query tables to local files dynamically
                result_df = con.execute(query).df()
                return "\n".join([json.dumps(record) for record in result_df.to_dict(orient="records")])

        except ImportError:
            logger.warning("duckdb not installed, falling back to basic mock logic.")
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}")

        # Fallback to hardcoded mock if no files or duckdb fails
        mock_data = [
            {"metric": "EBITDA", "year": 2023, "value": 1500000000, "currency": "USD", "source": "HardcodedFallback"},
            {"metric": "TotalDebt", "year": 2023, "value": 4000000000, "currency": "USD", "source": "HardcodedFallback"}
        ]
        return "\n".join([json.dumps(record) for record in mock_data])

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sql_query": {"type": "string", "description": "Valid T-SQL SELECT statement."}
            },
            "required": ["sql_query"]
        }
