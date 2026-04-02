import json
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import datetime as dt_module
import duckdb
import pandas as pd
from core.schemas.hnasp import HNASPState as HNASP

class ObservationLakehouse:
    """
    An Observation Lakehouse implemented via DuckDB and Apache Parquet.
    Stores Stimulus-Response Cubes (SRCs) to deterministically capture
    environmental actuation, contextual state, tool invocations, and generated outputs.
    """
    def __init__(self, storage_path: str = "data/lakehouse"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.db_path = os.path.join(storage_path, "lakehouse.duckdb")

        # Initialize DuckDB connection
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        """Initializes the Parquet-backed table structure."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id VARCHAR PRIMARY KEY,
                agent_id VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                state_data JSON NOT NULL
            )
        """)

    def save_trace(self, hnasp: HNASP):
        """
        Appends the HNASP state as a new row (trace) in DuckDB, which can be exported to Parquet.
        """
        trace_id = str(uuid.uuid4())
        agent_id = hnasp.meta.agent_id
        timestamp = datetime.now(timezone.utc)
        state_data = hnasp.model_dump_json(by_alias=True)

        # Insert into DuckDB table
        self.conn.execute(
            "INSERT INTO traces (trace_id, agent_id, timestamp, state_data) VALUES (?, ?, ?, ?)",
            [trace_id, agent_id, timestamp, state_data]
        )

        # Note: In an enterprise environment, Parquet export should be handled asynchronously or via batching,
        # rather than on every single trace insertion to avoid severe disk I/O and latency bottlenecks.
        # Parquet export is deferred to manual analytical triggers or scheduled tasks.

    def load_latest_state(self, agent_id: str) -> Optional[HNASP]:
        """
        Loads the most recent state for the given agent from DuckDB.
        """
        result = self.conn.execute("""
            SELECT state_data FROM traces
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, [agent_id]).fetchone()

        if result:
            state_data = result[0]
            try:
                # Assuming state_data is already a string of JSON in DuckDB based on our schema type JSON
                if not isinstance(state_data, str):
                    state_data = str(state_data)
                data = json.loads(state_data)
                return HNASP.model_validate(data)
            except Exception as e:
                print(f"Error loading state for {agent_id}: {e}")
                return None
        return None

    def query_traces(self, agent_id: str, limit: int = 10) -> List[HNASP]:
        """
        Retrieves the last N traces for sub-100ms behavioral clustering and auditing.
        """
        results = self.conn.execute("""
            SELECT state_data FROM traces
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, [agent_id, limit]).fetchall()

        traces = []
        # Reverse to return in chronological order
        for result in reversed(results):
            state_data = result[0]
            try:
                if not isinstance(state_data, str):
                    state_data = str(state_data)
                traces.append(HNASP.model_validate(json.loads(state_data)))
            except Exception:
                continue
        return traces

    def query_analytical(self, query: str) -> pd.DataFrame:
        """
        Executes arbitrary analytical queries against the Lakehouse.
        Returns a Pandas DataFrame for further Python processing.
        """
        return self.conn.execute(query).df()

    def close(self):
        """Closes the DuckDB connection."""
        self.conn.close()
