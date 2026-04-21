import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("EphemeralCortex")

class EphemeralCortex:
    """
    An in-memory contextual store for the Neural Mesh.
    Allows agents to query historical context or subscribe to specific topics
    rather than just relying on active broadcast connections.
    """
    def __init__(self, max_history_per_topic: int = 100):
        self.max_history = max_history_per_topic
        # Map of topic -> List of NeuralPacket payloads
        self._memory_bank: Dict[str, List[Any]] = {}
        self._lock = asyncio.Lock()

    async def ingest(self, packet_type: str, packet: Any):
        """Stores a packet in the short-term memory bank by topic."""
        async with self._lock:
            if packet_type not in self._memory_bank:
                self._memory_bank[packet_type] = []

            self._memory_bank[packet_type].append(packet)

            # Prune old memories
            if len(self._memory_bank[packet_type]) > self.max_history:
                self._memory_bank[packet_type].pop(0)

    async def query(self, packet_type: str, limit: int = 10) -> List[Any]:
        """Retrieves recent context for a specific topic."""
        async with self._lock:
            history = self._memory_bank.get(packet_type, [])
            return history[-limit:]

    async def flush(self):
        """Clears the ephemeral memory."""
        async with self._lock:
            self._memory_bank.clear()
            logger.info("Ephemeral Cortex flushed.")
