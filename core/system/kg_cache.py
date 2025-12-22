# core/system/kg_cache.py

try:
    import redis
except ImportError:
    redis = None


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
        self.ttl = ttl
        self.mock_cache = {}
        self.redis = None
        if redis:
            try:
                self.redis = redis.Redis(host=host, port=port, db=db)
                self.redis.ping()  # Check connection
            except Exception:
                self.redis = None  # Fallback to mock

    def get(self, query: str) -> str:
        """
        Gets the result of a SPARQL query from the cache.

        Args:
            query: The SPARQL query.

        Returns:
            The cached result, or None if not found.
        """
        if self.redis:
            try:
                return self.redis.get(query)
            except Exception:
                pass
        return self.mock_cache.get(query)

    def set(self, query: str, result: str):
        """
        Sets the result of a SPARQL query in the cache.

        Args:
            query: The SPARQL query.
            result: The result of the query.
        """
        if self.redis:
            try:
                self.redis.setex(query, self.ttl, result)
            except Exception:
                pass
        self.mock_cache[query] = result
