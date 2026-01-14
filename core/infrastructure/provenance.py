import json
import time
import uuid
import os
from typing import Dict, Any, Optional

class ProvenanceLogger:
    """
    Logs data lineage and execution context for model runs.
    Ensures traceability of AI decision making and simulation outputs.
    """
    def __init__(self, log_file: str = "data/provenance_log.jsonl"):
        self.log_file = log_file
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_execution(self,
                      module: str,
                      operation: str,
                      inputs: Dict[str, Any],
                      outputs: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Records an execution event.
        """
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "module": module,
            "operation": operation,
            "inputs": self._sanitize(inputs),
            "outputs": self._sanitize(outputs),
            "metadata": metadata or {},
            "environment": {
                "user": os.getenv("USER", "unknown"),
                "session_id": os.getenv("JULES_SESSION_ID", "unknown")
            }
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _sanitize(self, data: Any) -> Any:
        """
        Sanitizes data for logging (prevents massive dumps).
        Truncates lists/dicts if they are too large.
        """
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, list):
            if len(data) > 10:
                return [self._sanitize(i) for i in data[:10]] + [f"... ({len(data)-10} more items)"]
            return [self._sanitize(i) for i in data]
        elif hasattr(data, "model_dump"): # Pydantic v2
            return self._sanitize(data.model_dump())
        elif hasattr(data, "dict"): # Pydantic v1
            return self._sanitize(data.dict())
        else:
            return data
