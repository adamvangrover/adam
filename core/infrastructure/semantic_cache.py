import hashlib
import json
import os
import time
from typing import Dict, Any, Optional

class SemanticCache:
    """
    Implements a Semantic Cache with Provenance awareness.
    Caches outputs based on inputs + logic version + underlying data hash.
    """

    def __init__(self, cache_dir: str = "core/libraries_and_archives/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_key(self, prompt: str, context_hash: str, model_id: str) -> str:
        """Generates a deterministic cache key."""
        raw_key = f"{prompt}|{context_hash}|{model_id}"
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def get(self, prompt: str, context_hash: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached response if it exists and is valid."""
        key = self._generate_key(prompt, context_hash, model_id)
        path = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    entry = json.load(f)

                # Check expiry (optional, e.g. 24h)
                if time.time() - entry['timestamp'] > 86400:
                    return None

                return entry['output']
            except Exception:
                return None
        return None

    def set(self, prompt: str, context_hash: str, model_id: str, output: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Writes a response to the cache."""
        key = self._generate_key(prompt, context_hash, model_id)
        path = os.path.join(self.cache_dir, f"{key}.json")

        entry = {
            "timestamp": time.time(),
            "key": key,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "context_hash": context_hash,
            "model_id": model_id,
            "output": output,
            "metadata": metadata or {}
        }

        with open(path, 'w') as f:
            json.dump(entry, f, indent=2)

    @staticmethod
    def compute_data_hash(data: Any) -> str:
        """Helper to compute hash of arbitrary input data (dict, list, str)."""
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        else:
            # Sort keys for deterministic JSON
            encoded = json.dumps(data, sort_keys=True).encode()
            return hashlib.sha256(encoded).hexdigest()
