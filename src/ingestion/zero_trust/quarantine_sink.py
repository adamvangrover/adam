import json
import time
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class QuarantineSink:
    """
    Phase 4: Air-Gapped Quarantine Sink
    All payloads that fail TLS, Checksum, Consensus, or Schema validation are serialized
    and dumped to an isolated air-gapped logging sink/directory for forensic analysis.
    They are NOT processed in the main event loop.
    """
    def __init__(self, quarantine_dir: str = "logs/quarantine"):
        self.quarantine_dir = quarantine_dir
        os.makedirs(self.quarantine_dir, exist_ok=True)
        self.in_memory_log: List[Dict[str, Any]] = []

    def quarantine_payload(self, payload: Any, failure_reason: str, error_type: str, metadata: Dict[str, Any] = None) -> str:
        timestamp = time.time()
        record = {
            "timestamp": timestamp,
            "error_type": error_type,
            "failure_reason": failure_reason,
            "metadata": metadata or {},
            "payload": payload if isinstance(payload, (dict, list, str, int, float, bool)) else str(payload)
        }

        self.in_memory_log.append(record)

        # Write to air-gapped disk forensic sink
        filename = f"quarantine_{int(timestamp * 1000)}_{error_type}.json"
        filepath = os.path.join(self.quarantine_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2)
            logger.warning(f"AIR-GAPPED QUARANTINE: Payload dumped to {filepath}. Reason: {failure_reason}")
        except Exception as e:
            logger.error(f"Failed to write quarantine payload to disk: {e}")

        return filepath

    def get_quarantined_records(self) -> List[Dict[str, Any]]:
        return self.in_memory_log
