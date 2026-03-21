import time
import logging

class PheromoneDB:
    """
    A lightweight, TTL-based memory cache for System 1 swarm agents.
    """
    def __init__(self, ttl: int = 60):
        self._db = {}
        self._ttl = ttl
        self.logger = logging.getLogger(self.__class__.__name__)

    def set(self, key: str, value: any):
        """
        Set a value with TTL.
        """
        self._db[key] = {
            'value': value,
            'expires_at': time.time() + self._ttl
        }
        self.logger.debug(f"Pheromone set: {key}")

    def get(self, key: str):
        """
        Get a value. Returns None if expired or not found.
        """
        if key in self._db:
            record = self._db[key]
            if record['expires_at'] > time.time():
                self.logger.debug(f"Pheromone hit: {key}")
                return record['value']
            else:
                self.logger.debug(f"Pheromone expired: {key}")
                del self._db[key]
        return None

    def clear(self):
        self._db.clear()
