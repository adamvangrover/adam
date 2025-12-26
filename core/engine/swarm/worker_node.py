# core/engine/swarm/worker_node.py

"""
SwarmWorker: The base unit of the Hive Mind.
Designed for massive parallelism, minimal state, and specific task execution.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional
from core.engine.swarm.pheromone_board import PheromoneBoard

logger = logging.getLogger(__name__)

class SwarmWorker:
    def __init__(self, board: PheromoneBoard, role: str = "generalist"):
        self.id = str(uuid.uuid4())[:8]
        self.board = board
        self.role = role
        self.is_active = True
        logger.info(f"SwarmWorker {self.id} ({self.role}) initialized.")

    async def run(self):
        """
        Main loop for the worker.
        """
        while self.is_active:
            # 1. Look for work (sniff pheromones)
            tasks = await self.board.sniff(signal_type=f"TASK_{self.role.upper()}")

            if tasks:
                # Pick the highest intensity task
                tasks.sort(key=lambda x: x.intensity, reverse=True)
                task_signal = tasks[0]

                # Try to claim it (consume)
                # In a real distributed system, this needs better locking, but for async locally this is okay-ish
                await self.board.consume(task_signal)

                # Execute
                await self.execute_task(task_signal.data)
            else:
                # No work, wait a bit
                await asyncio.sleep(1)

    async def execute_task(self, data: Dict[str, Any]):
        """
        Execute the specific logic. Override this in subclasses.
        """
        logger.info(f"Worker {self.id} executing task: {data}")

        # Simulate work
        await asyncio.sleep(0.5)

        # Report success via pheromone
        result_data = {"status": "success", "original_task": data, "worker": self.id}
        await self.board.deposit("RESULT", result_data, intensity=10.0, source=self.id)

    def stop(self):
        self.is_active = False

class AnalysisWorker(SwarmWorker):
    async def execute_task(self, data: Dict[str, Any]):
        target = data.get("target")
        logger.info(f"Worker {self.id} analyzing {target}...")

        # Mock analysis
        await asyncio.sleep(1.0)
        score = len(target) * 10 if target else 0

        result = {
            "target": target,
            "risk_score": score,
            "analyst_comment": f"Analyzed by {self.id}"
        }

        await self.board.deposit("ANALYSIS_RESULT", result, intensity=10.0, source=self.id)
