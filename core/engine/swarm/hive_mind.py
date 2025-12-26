# core/engine/swarm/hive_mind.py

"""
HiveMind: The orchestrator for the Swarm.
Manages the PheromoneBoard and the lifecycle of SwarmWorkers.
"""

import asyncio
import logging
from typing import List, Dict, Any
from core.engine.swarm.pheromone_board import PheromoneBoard
from core.engine.swarm.worker_node import SwarmWorker, AnalysisWorker

logger = logging.getLogger(__name__)

class HiveMind:
    def __init__(self, worker_count: int = 5):
        self.board = PheromoneBoard()
        self.workers: List[SwarmWorker] = []
        self.worker_count = worker_count
        self.tasks = []

    async def initialize(self):
        """
        Spin up the workers.
        """
        logger.info(f"HiveMind initializing with {self.worker_count} workers...")
        for _ in range(self.worker_count):
            worker = AnalysisWorker(self.board, role="analyst")
            self.workers.append(worker)
            self.tasks.append(asyncio.create_task(worker.run()))

    async def disperse_task(self, task_type: str, payload: Dict[str, Any], intensity: float = 10.0):
        """
        Broadcast a task to the swarm.
        """
        logger.info(f"HiveMind dispersing task: {task_type}")
        await self.board.deposit(f"TASK_{task_type.upper()}", payload, intensity=intensity, source="HiveMind")

    async def gather_results(self, signal_type: str = "ANALYSIS_RESULT", timeout: float = 5.0) -> List[Dict]:
        """
        Wait for and collect results.
        """
        logger.info("HiveMind gathering results...")
        start_time = asyncio.get_event_loop().time()
        results = []

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            pheromones = await self.board.sniff(signal_type)
            for p in pheromones:
                results.append(p.data)
                await self.board.consume(p) # Mark as collected

            if len(results) >= 1: # For demo, return as soon as we have something, or wait for condition
                # In real scenario, we might wait for N results
                pass

            await asyncio.sleep(0.5)

        return results

    async def shutdown(self):
        """
        Stop all workers.
        """
        logger.info("HiveMind shutting down...")
        for w in self.workers:
            w.stop()

        # Cancel asyncio tasks
        for t in self.tasks:
            t.cancel()
