import networkx as nx
import logging
import threading

logger = logging.getLogger(__name__)

class GraphCache:
    """
    Thread-safe Singleton Cache for the Unified Knowledge Graph.
    Ensures the graph is loaded only once per process.
    """
    _instance = None
    _lock = threading.Lock()
    _graph = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_graph(self) -> nx.DiGraph:
        """Returns the shared graph instance."""
        return self._graph

    def set_graph(self, graph: nx.DiGraph):
        """Sets the shared graph instance. Should be called only once."""
        with self._lock:
            if self._graph is None:
                self._graph = graph
                logger.info("GraphCache: Shared graph instance set.")
            else:
                logger.debug("GraphCache: Graph already set. Ignoring update.")

    def clear(self):
        """Clears the cache (for testing purposes)."""
        with self._lock:
            self._graph = None
            logger.info("GraphCache: Cache cleared.")
