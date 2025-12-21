import logging
import pickle
from typing import Any, Dict, Optional

# Conditional import to handle missing dependencies safely
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages persistence of optimizer states (momentum, variance, step count).
    Uses Redis for production, falls back to in-memory for testing/local dev.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = None
        self.local_store = {}
        self.using_redis = False

        if REDIS_AVAILABLE:
            try:
                self.redis = redis.from_url(redis_url, socket_connect_timeout=1)
                self.redis.ping()
                self.using_redis = True
                logger.info("Connected to Redis for state persistence.")
            except (redis.ConnectionError, redis.TimeoutError):
                logger.warning("Redis not reachable. Falling back to in-memory storage.")
        else:
             logger.warning("Redis library not installed. Falling back to in-memory storage.")

    def save_state(self, key: str, state: Dict[str, Any]):
        """Save optimizer state dictionary."""
        if self.using_redis and self.redis:
            try:
                # Pickle allows us to store torch tensors directly
                data = pickle.dumps(state)
                self.redis.set(key, data)
            except Exception as e:
                logger.error(f"Failed to save state to Redis: {e}")
                # Fallback to local
                self.local_store[key] = state
        else:
            self.local_store[key] = state

    def load_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Load optimizer state dictionary."""
        if self.using_redis and self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return pickle.loads(data)
            except Exception as e:
                logger.error(f"Failed to load state from Redis: {e}")
                return self.local_store.get(key)

        return self.local_store.get(key)
