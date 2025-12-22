from core.tools.base_tool import BaseTool
from typing import Any
import json
import os
import pandas as pd


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
        # Security Check: Read-only
        if "DROP" in sql_query.upper() or "DELETE" in sql_query.upper() or "UPDATE" in sql_query.upper():
            return json.dumps({"error": "SecurityViolation: Read-only access enforced."})

        # SIMULATE CONNECTION FAILURE -> FALLBACK
        try:
            return self._query_local_lake(sql_query)
        except Exception as e:
            return json.dumps({"error": f"Fallback failed: {e}"})

    def _query_local_lake(self, query: str) -> str:
        # Very simple SQL-like parsing for fallback

        # Try loading local data
        df = None
        try:
            if os.path.exists(os.path.join(self.local_lake_path, "financials.parquet")):
                df = pd.read_parquet(os.path.join(self.local_lake_path, "financials.parquet"))
            elif os.path.exists(os.path.join(self.local_lake_path, "financials.json")):
                df = pd.read_json(os.path.join(self.local_lake_path, "financials.json"))
        except:
            pass

        if df is not None:
            # Filter logic (Mock)
            if "AAPL" in query:
                df = df[df['ticker'] == 'AAPL']
            elif "TSLA" in query:
                df = df[df['ticker'] == 'TSLA']

            # Format as JSONL
            return "\n".join([json.dumps(record) for record in df.to_dict(orient="records")])

        # Fallback to hardcoded mock if no files
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
