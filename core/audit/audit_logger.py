import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AuditLog(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    user_id: str

    # Model State
    prompt_version_id: str
    model_id: str
    hyperparameters: Dict[str, Any]

    # Context
    retrieved_chunks: List[Dict[str, Any]]
    graph_context: List[Any]

    # Output
    raw_output: str
    citations: List[str]

    # Safety
    guardrail_status: str = "PASS"

    # Outcome (Optional for now)
    human_edit_diff: Optional[str] = None

class AuditLogger:
    def __init__(self, log_path: str = "data/audit_log.jsonl"):
        self.log_path = log_path

    def log_event(self, log_entry: AuditLog):
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_entry.model_dump_json() + "\n")
        except Exception as e:
            logging.error(f"Failed to write to audit log: {e}")

    def get_logs(self) -> List[AuditLog]:
        logs = []
        try:
            with open(self.log_path, 'r') as f:
                for line in f:
                    logs.append(AuditLog.parse_raw(line))
        except FileNotFoundError:
            pass
        return logs
