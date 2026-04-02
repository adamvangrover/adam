import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any


class SemanticCache:
    """
    Implements a Semantic Cache with Provenance awareness.
    Caches outputs based on inputs + logic version + underlying data hash.
    Employs a two-tier hybrid caching strategy: an in-memory OrderedDict LRU cache
    to prevent disk I/O bottlenecks on repeated hits, layered above a persistent
    file-based JSON disk cache.
    """

    def __init__(self, cache_dir: str = "core/libraries_and_archives/cache", memory_capacity: int = 1000):
        self.cache_dir = cache_dir
        self.memory_capacity = memory_capacity
        self._memory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_key(self, prompt: str, context_hash: str, model_id: str) -> str:
        """Generates a deterministic cache key."""
        raw_key = f"{prompt}|{context_hash}|{model_id}"
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def get(self, prompt: str, context_hash: str, model_id: str) -> dict[str, Any] | None:
        """Retrieves a cached response if it exists and is valid."""
        key = self._generate_key(prompt, context_hash, model_id)

        # 1. Check in-memory LRU cache
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() - entry['timestamp'] > 86400:
                self._memory_cache.pop(key, None)
                return None
            self._memory_cache.move_to_end(key)
            return entry['output']

        # 2. Fallback to disk cache
        path = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    entry = json.load(f)

                # Check expiry (e.g., 24h)
                if time.time() - entry['timestamp'] > 86400:
                    return None

                # Update memory cache
                self._memory_cache[key] = entry
                if len(self._memory_cache) > self.memory_capacity:
                    self._memory_cache.popitem(last=False)

                return entry['output']
            except Exception:
                return None
        return None

    def set(self, prompt: str, context_hash: str, model_id: str, output: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
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

        # Update memory cache
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        self._memory_cache[key] = entry

        # Enforce LRU policy on memory cache
        while len(self._memory_cache) > self.memory_capacity:
            self._memory_cache.popitem(last=False)

        with open(path, 'w') as f:
            json.dump(entry, f, indent=2)

    def clear(self) -> None:
        """Clears the in-memory cache and optionally could remove disk files."""
        self._memory_cache.clear()

    @staticmethod
    def compute_data_hash(data: Any) -> str:
        """Helper to compute hash of arbitrary input data (dict, list, str)."""
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()
        else:
            # Sort keys for deterministic JSON
            encoded = json.dumps(data, sort_keys=True).encode()
            return hashlib.sha256(encoded).hexdigest()
