"""
Provenance Logger for Adam v23.5 System.
Implements a lightweight W3C PROV-O compliant logging mechanism to ensure auditability of agent decisions.
"""

import json
import uuid
import datetime
import os
from typing import Dict, Any, Optional
from enum import Enum

class ActivityType(str, Enum):
    GENERATION = "generation"
    REVIEW = "review"
    SIMULATION = "simulation"
    MODIFICATION = "modification"
    DECISION = "decision"

class ProvenanceLogger:
    def __init__(self, log_dir: str = "core/libraries_and_archives/audit_trails/"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())

    def log_activity(self,
                     agent_id: str,
                     activity_type: ActivityType,
                     input_data: Dict[str, Any],
                     output_data: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Logs a specific activity with provenance metadata.
        Returns the unique Record ID.
        """
        record_id = f"PROV-{uuid.uuid4()}"
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        entry = {
            "record_id": record_id,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "agent_id": agent_id,
            "activity_type": activity_type,
            "entity_context": {
                "inputs_hash": str(hash(json.dumps(input_data, sort_keys=True))),
                "inputs_sample": str(input_data)[:200]  # Truncate for brevity in log overview
            },
            "outcome": {
                "status": "SUCCESS", # Default, could be extended
                "output_keys": list(output_data.keys())
            },
            "metadata": metadata or {}
        }

        # In a real PROV-O RDF system, we would map these to prov:Activity, prov:Agent, etc.
        # Here we use a JSON-LD friendly structure.

        filename = f"{self.log_dir}{record_id}.json"
        with open(filename, 'w') as f:
            json.dump(entry, f, indent=2)

        return record_id

    def get_session_history(self) -> list:
        """Retrieves all logs for the current session."""
        history = []
        for filename in os.listdir(self.log_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, filename), 'r') as f:
                        data = json.load(f)
                        if data.get("session_id") == self.session_id:
                            history.append(data)
                except Exception:
                    continue
        return sorted(history, key=lambda x: x['timestamp'])

if __name__ == "__main__":
    # Test
    logger = ProvenanceLogger()
    logger.log_activity("TestAgent", ActivityType.DECISION, {"query": "Buy PLTR?"}, {"decision": "BUY"})
    print(f"Logged session {logger.session_id}")
