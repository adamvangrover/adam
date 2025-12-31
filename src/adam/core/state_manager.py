import pickle
import logging
import sys
import os
from typing import Dict, Any, Optional

# Conditional import to handle missing dependencies safely
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import Secure Unpickler
# We attempt to import from the core package.
# Depending on the execution environment, 'core' might be in the python path.
try:
    from core.security.safe_unpickler import safe_loads
except ImportError:
    # If standard import fails, try to look up from root if we are deep in src
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
        from core.security.safe_unpickler import safe_loads
    except ImportError:
        # 🛡️ Sentinel: Fail Closed. Do not allow unsafe unpickling.
        safe_loads = None

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

        if self.using_redis and safe_loads is None:
             logger.error("Security Module 'core.security.safe_unpickler' not found. Redis persistence disabled to prevent RCE.")
             self.using_redis = False
             self.redis = None


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
                    if safe_loads:
                        # 🛡️ Sentinel: Use safe_loads to prevent RCE
                        return safe_loads(data)
                    else:
                        logger.error("Safe loads unavailable. Cannot deserialize.")
                        return None
            except Exception as e:
                logger.error(f"Failed to load state from Redis: {e}")
                return self.local_store.get(key)

        return self.local_store.get(key)
