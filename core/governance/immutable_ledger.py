import json
import hashlib
import time
import os
import fcntl
from typing import Dict, Any

class ImmutableLedger:
    """
    A persistent, append-only log where each entry is cryptographically linked
    to the previous one via SHA256 hash chaining.
    """
    def __init__(self, ledger_path="data/immutable_ledger.jsonl"):
        self.ledger_path = ledger_path
        # Ensure dir exists
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)

    def _get_last_hash(self) -> str:
        """Reads the last line of the ledger to get the previous hash."""
        last_hash = "0" * 64 # Genesis hash
        if not os.path.exists(self.ledger_path):
            return last_hash

        try:
            # Read last line efficiently (seek to end and scan back?)
            # For simplicity in MVP, we assume file isn't massive yet or read lines
            with open(self.ledger_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    try:
                        last_entry = json.loads(lines[-1])
                        last_hash = last_entry.get("current_hash", last_hash)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return last_hash

    def log_entry(self, actor: str, action: str, details: Dict[str, Any]) -> str:
        """
        Appends a new entry to the ledger.
        """
        prev_hash = self._get_last_hash()
        timestamp = time.time()

        # Payload to hash
        payload = {
            "timestamp": timestamp,
            "actor": actor,
            "action": action,
            "details": details,
            "previous_hash": prev_hash
        }

        # Calculate Hash
        # Sort keys to ensure deterministic hashing
        payload_str = json.dumps(payload, sort_keys=True)
        current_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        final_entry = payload
        final_entry["current_hash"] = current_hash

        # Append to file with locking
        with open(self.ledger_path, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(final_entry) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        return current_hash
