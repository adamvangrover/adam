import time
from typing import Any, Dict, Optional

class PheromoneDB:
    """
    TTL-based PheromoneDB caching mechanism for System 1 Swarm memory.
    Provides non-blocking access to real-time state parameters.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl: int = 60) -> None:
        """
        Stores a value in the cache with a specified Time-To-Live (TTL) in seconds.
        """
        expires_at = time.time() + ttl
        self._cache[key] = {
            "value": value,
            "expires_at": expires_at
        }

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the cache if it hasn't expired.
        If expired, it is removed and None is returned.
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def cleanup(self) -> None:
        """
        Actively removes all expired keys from the cache.
        """
        current_time = time.time()
        keys_to_delete = [
            key for key, entry in self._cache.items()
            if current_time > entry["expires_at"]
        ]
        for key in keys_to_delete:
            del self._cache[key]
