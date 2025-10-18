# core/system/kg_cache.py

import redis

class KGCache:
    """
    A caching layer for SPARQL queries using Redis.
    """

    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        """
        Initializes the KGCache.

        Args:
            host: The Redis host.
            port: The Redis port.
            db: The Redis database.
            ttl: The Time-To-Live for cache entries in seconds.
        """
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.ttl = ttl

    def get(self, query: str) -> str:
        """
        Gets the result of a SPARQL query from the cache.

        Args:
            query: The SPARQL query.

        Returns:
            The cached result, or None if not found.
        """
        return self.redis.get(query)

    def set(self, query: str, result: str):
        """
        Sets the result of a SPARQL query in the cache.

        Args:
            query: The SPARQL query.
            result: The result of the query.
        """
        self.redis.setex(query, self.ttl, result)
