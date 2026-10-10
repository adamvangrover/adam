from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()

@dataclass(frozen=True)
class ProvenanceRecord:
    event: dict[str, Any]
    previous_hash: str | None = None

    def digest(self) -> str:
        payload = {
            "previous_hash": self.previous_hash,
            "event": self.event,
        }
        return sha256(canonical_json(payload))

    def as_dict(self) -> dict[str, Any]:
        digest = self.digest()
        return {
            "previous_hash": self.previous_hash,
            "event": self.event,
            "record_hash": digest,
        }
