import json
import os
from typing import List, Optional, Dict
from datetime import datetime
from core.schemas.hnasp import HNASP

class ObservationLakehouse:
    def __init__(self, storage_path: str = "data/lakehouse"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def _get_agent_path(self, agent_id: str) -> str:
        return os.path.join(self.storage_path, f"{agent_id}.jsonl")

    def save_trace(self, hnasp: HNASP):
        """
        Appends the HNASP state as a new row (trace) in the JSONL file.
        """
        path = self._get_agent_path(hnasp.meta.agent_id)
        with open(path, "a") as f:
            f.write(hnasp.model_dump_json(by_alias=True) + "\n")

    def load_latest_state(self, agent_id: str) -> Optional[HNASP]:
        """
        Loads the most recent state for the given agent.
        """
        path = self._get_agent_path(agent_id)
        if not os.path.exists(path):
            return None

        last_line = None
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    last_line = line

        if last_line:
            try:
                data = json.loads(last_line)
                return HNASP.model_validate(data)
            except Exception as e:
                print(f"Error loading state for {agent_id}: {e}")
                return None
        return None

    def query_traces(self, agent_id: str, limit: int = 10) -> List[HNASP]:
        """
        Retrieves the last N traces.
        """
        path = self._get_agent_path(agent_id)
        if not os.path.exists(path):
            return []

        traces = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    traces.append(HNASP.model_validate(json.loads(line)))
                except:
                    continue
        return traces
