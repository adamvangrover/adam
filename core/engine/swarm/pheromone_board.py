# core/engine/swarm/pheromone_board.py

"""
PheromoneBoard: A shared blackboard for the Swarm.
Allows agents to deposit 'signals' (pheromones) that decay over time or are consumed by other agents.
This implements the 'Stigmergy' coordination pattern.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Pheromone:
    signal_type: str
    intensity: float
    data: Dict[str, Any]
    source_agent_id: str
    timestamp: float = field(default_factory=time.time)
    decay_rate: float = 0.1  # Units per minute

class PheromoneBoard:
    def __init__(self):
        self._pheromones: Dict[str, List[Pheromone]] = {}
        self._lock = asyncio.Lock()

    async def deposit(self, signal_type: str, data: Dict[str, Any], intensity: float = 1.0, source: str = "unknown"):
        """
        Deposit a new pheromone signal.
        """
        async with self._lock:
            if signal_type not in self._pheromones:
                self._pheromones[signal_type] = []

            pheromone = Pheromone(
                signal_type=signal_type,
                intensity=intensity,
                data=data,
                source_agent_id=source
            )
            self._pheromones[signal_type].append(pheromone)
            logger.debug(f"Pheromone deposited: {signal_type} by {source}")

    async def sniff(self, signal_type: str, min_intensity: float = 0.1) -> List[Pheromone]:
        """
        Retrieve active pheromones of a certain type.
        """
        current_time = time.time()
        active_pheromones = []

        async with self._lock:
            if signal_type in self._pheromones:
                # Filter and decay
                valid_pheromones = []
                for p in self._pheromones[signal_type]:
                    # Calculate current intensity
                    age_minutes = (current_time - p.timestamp) / 60.0
                    current_intensity = p.intensity - (p.decay_rate * age_minutes)

                    if current_intensity >= min_intensity:
                        p.intensity = current_intensity # Update state
                        valid_pheromones.append(p)
                        active_pheromones.append(p)

                # Cleanup decayed signals
                self._pheromones[signal_type] = valid_pheromones

        return active_pheromones

    async def consume(self, pheromone: Pheromone):
        """
        Remove a pheromone after processing it.
        """
        async with self._lock:
            if pheromone.signal_type in self._pheromones:
                try:
                    self._pheromones[pheromone.signal_type].remove(pheromone)
                    logger.debug(f"Pheromone consumed: {pheromone.signal_type}")
                except ValueError:
                    pass
