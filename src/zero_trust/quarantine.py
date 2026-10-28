"""
Air-gapped Quarantine Matrix Logging Sink.
All payloads failing TLS, Checksum, Consensus, or Schema validation are serialized
and dumped here for forensic analysis without being processed in the main event loop.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

QUARANTINE_DIR = "logs/quarantine"
os.makedirs(QUARANTINE_DIR, exist_ok=True)


class QuarantineMatrix:
    """
    Air-gapped forensic logging sink for rejected/poisoned payloads.
    """

    def __init__(self, storage_dir: str = QUARANTINE_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.quarantine_memory: List[Dict[str, Any]] = []

    def quarantine_payload(
        self,
        reason: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Serializes and dumps failed payload into air-gapped forensic quarantine sink.
        """
        quarantine_id = f"Q-{int(time.time() * 1000)}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        entry = {
            "quarantine_id": quarantine_id,
            "timestamp": timestamp,
            "rejection_reason": reason,
            "headers": headers or {},
            "client_info": client_info or {},
            "raw_payload": payload,
        }

        file_path = os.path.join(self.storage_dir, f"{quarantine_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to dump quarantined payload to file {file_path}: {e}")

        self.quarantine_memory.append(entry)
        logger.critical(f"PAYLOAD QUARANTINED [{quarantine_id}]: {reason}")
        return entry

    def get_quarantined_records(self) -> List[Dict[str, Any]]:
        """Returns all quarantined forensic records."""
        return self.quarantine_memory
