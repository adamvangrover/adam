# core/engine/swarm/hive_mind.py

"""
HiveMind: The orchestrator for the Swarm.
Manages the PheromoneBoard and the lifecycle of SwarmWorkers.
"""

import asyncio
import logging
from typing import List, Dict, Any
from core.engine.swarm.pheromone_board import PheromoneBoard
from core.engine.swarm.worker_node import SwarmWorker, AnalysisWorker, CoderWorker, ReviewerWorker, TesterWorker

logger = logging.getLogger(__name__)

class HiveMind:
    def __init__(self, worker_count: int = 5):
        self.board = PheromoneBoard()
        self.workers: List[SwarmWorker] = []
        self.worker_count = worker_count
        self.tasks = []

    async def initialize(self):
        """
        Spin up the workers with a diverse distribution of roles.
        """
        logger.info(f"HiveMind initializing with {self.worker_count} workers...")

        # Determine distribution
        # e.g., 40% Analyst, 20% Coder, 20% Reviewer, 20% Tester
        count_analyst = max(1, int(self.worker_count * 0.4))
        count_coder = int(self.worker_count * 0.2)
        count_reviewer = int(self.worker_count * 0.2)
        count_tester = self.worker_count - count_analyst - count_coder - count_reviewer

        # Spawn Analysts
        for _ in range(count_analyst):
            self._spawn_worker(AnalysisWorker, "analyst")

        # Spawn Coders
        for _ in range(count_coder):
            self._spawn_worker(CoderWorker, "coder")

        # Spawn Reviewers
        for _ in range(count_reviewer):
            self._spawn_worker(ReviewerWorker, "reviewer")

        # Spawn Testers
        for _ in range(count_tester):
            self._spawn_worker(TesterWorker, "tester")

    def _spawn_worker(self, worker_class, role):
        worker = worker_class(self.board, role=role)
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
