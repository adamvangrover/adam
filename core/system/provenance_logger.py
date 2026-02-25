"""
Provenance Logger for Adam v26.0 System.
Implements a lightweight W3C PROV-O compliant logging mechanism to ensure auditability of agent decisions.
"""

import json
import uuid
import datetime
import os
import hashlib
import psutil
import subprocess
from typing import Dict, Any, Optional
from enum import Enum

class ActivityType(str, Enum):
    GENERATION = "generation"
    REVIEW = "review"
    SIMULATION = "simulation"
    MODIFICATION = "modification"
    DECISION = "decision"
    INGESTION = "ingestion"
    CALCULATION = "calculation"

class ProvenanceLogger:
    def __init__(self, log_dir: str = "core/libraries_and_archives/audit_trails/", use_jsonl: bool = True):
        self.log_dir = log_dir
        self.use_jsonl = use_jsonl
        self.jsonl_path = os.path.join(log_dir, "provenance_stream.jsonl")
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        self.git_revision = self._get_git_revision()
        self.version = "23.5.0" # Should match setup.py/pyproject.toml

    def _get_git_revision(self):
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "unknown"

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        try:
            dump = json.dumps(data, sort_keys=True)
            return hashlib.sha256(dump.encode('utf-8')).hexdigest()
        except TypeError:
            return str(hash(str(data)))

    def _get_compute_metrics(self) -> Dict[str, Any]:
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "process_memory_mb": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            }
        except Exception:
            return {}

    def log_activity(self,
                     agent_id: str,
                     activity_type: ActivityType,
                     input_data: Dict[str, Any],
                     output_data: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None,
                     trace_id: Optional[str] = None,
                     data_source: Optional[str] = None,
                     capture_full_io: bool = False) -> str:
        """
        Logs a specific activity with provenance metadata.
        Returns the unique Record ID.
        """
        record_id = f"PROV-{uuid.uuid4()}"
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        # Ensure activity_type is string
        act_type = activity_type.value if isinstance(activity_type, ActivityType) else str(activity_type)

        entry = {
            "record_id": record_id,
            "session_id": self.session_id,
            "trace_id": trace_id or self.session_id, # Default to session if no trace
            "timestamp": timestamp,
            "agent_id": agent_id,
            "activity_type": act_type,
            "build_provenance": {
                "version": self.version,
                "git_hash": self.git_revision
            },
            "compute_metrics": self._get_compute_metrics(),
            "data_lineage": {
                "source": data_source or "unknown",
                "input_hash": self._generate_hash(input_data),
                "output_hash": self._generate_hash(output_data)
            },
            "entity_context": {
                "inputs": input_data if capture_full_io else str(input_data)[:500],
                "outputs": output_data if capture_full_io else list(output_data.keys())
            },
            "metadata": metadata or {}
        }

        # JSONL Logging (Stream)
        if self.use_jsonl:
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")

        # Individual File Logging (Artifact)
        filename = f"{self.log_dir}{record_id}.json"
        with open(filename, 'w') as f:
            json.dump(entry, f, indent=2)

        return record_id

    def get_session_history(self) -> list:
        """Retrieves all logs for the current session."""
        history = []
        # Try reading from JSONL first if enabled
        if self.use_jsonl and os.path.exists(self.jsonl_path):
             with open(self.jsonl_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("session_id") == self.session_id:
                            history.append(data)
                    except json.JSONDecodeError:
                        continue
             return sorted(history, key=lambda x: x['timestamp'])

        # Fallback to individual files
        for filename in os.listdir(self.log_dir):
            if filename.endswith(".json") and filename != "provenance_stream.jsonl":
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
    logger.log_activity("TestAgent", ActivityType.DECISION, {"query": "Buy PLTR?"}, {"decision": "BUY"}, trace_id="TRACE-123", data_source="Bloomberg")
    print(f"Logged session {logger.session_id}")
