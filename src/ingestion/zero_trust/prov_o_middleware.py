import json
import hashlib
from typing import Dict, Any, Union

class ProvOChecksumError(Exception):
    """Raised when the computed SHA-256 payload checksum does not match the expected source checksum."""
    pass

class ProvOMiddleware:
    """
    W3C PROV-O Compliance & Hashing Middleware layer.
    Computes a SHA-256 hash of incoming JSON payloads and verifies against source checksums.
    Drops mismatched payloads immediately.
    """
    @staticmethod
    def compute_sha256(data: Union[bytes, str, Dict[str, Any]]) -> str:
        if isinstance(data, dict):
            # Canonical JSON representation for deterministic hashing
            raw_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif isinstance(data, str):
            raw_bytes = data.encode("utf-8")
        elif isinstance(data, bytes):
            raw_bytes = data
        else:
            raise ValueError(f"Unsupported payload type for checksum hashing: {type(data)}")

        return hashlib.sha256(raw_bytes).hexdigest()

    def process_payload(self, raw_payload: Union[bytes, str, Dict[str, Any]], expected_checksum: str) -> str:
        computed_hash = self.compute_sha256(raw_payload)

        if expected_checksum.lower() != computed_hash.lower():
            raise ProvOChecksumError(
                f"W3C PROV-O Cryptographic Verification Failed! "
                f"Expected: {expected_checksum}, Computed: {computed_hash}"
            )

        return computed_hash
