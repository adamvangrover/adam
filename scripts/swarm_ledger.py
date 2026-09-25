from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional
import hashlib
import json

class SwarmMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    message_id: str
    sender: str
    payload: dict
    timestamp: int

class AppendOnlyLedger(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    entries: List[SwarmMessage] = []

    def append(self, message: SwarmMessage):
        self.entries.append(message)

    def hash_cache(self) -> str:
        data = json.dumps([m.model_dump() for m in self.entries], sort_keys=True)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

class KVStateStore(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')
    store: Dict[str, dict] = {}

    def set_state(self, key: str, value: dict):
        self.store[key] = value

    def get_state(self, key: str) -> Optional[dict]:
        return self.store.get(key)
