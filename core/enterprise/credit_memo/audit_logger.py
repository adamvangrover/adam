import json
import logging
import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from .model import AuditLogEntry

class AuditLogger:
    """
    Handles secure audit logging for compliance.
    Protocol: Iron Bank
    """
    def __init__(self, log_dir: str = "core/data/audit_logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl")

    def log_event(self, entry: AuditLogEntry):
        """
        Logs a structured event to the audit trail.
        """
        try:
            log_data = entry.model_dump()
            # Append to JSONL file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data) + "\n")

            logging.info(f"AUDIT_LOG: Transaction {entry.transaction_id} recorded.")
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")

    def query_logs(self, transaction_id: str) -> Optional[AuditLogEntry]:
        """
        Retrieves a log entry by transaction ID (simple linear scan for simulation).
        """
        if not os.path.exists(self.log_file):
            return None

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("transaction_id") == transaction_id:
                        return AuditLogEntry(**data)
        except Exception as e:
            logging.error(f"Failed to query audit log: {e}")
        return None

# Global Instance
audit_logger = AuditLogger()
