import asyncio
import random
from typing import Any, Dict

from core.agents.agent_base import AgentBase
from core.agents.mixins.redundancy_optimization_mixin import RedundancyOptimizationMixin

class OptimizedGalleryAgent(AgentBase, RedundancyOptimizationMixin):
    """
    Demonstration agent that uses RedundancyOptimizationMixin to securely and efficiently
    fetch 'gallery data' (simulated) with fallback capabilities.
    """

    async def execute(self, task: str = "fetch_data", *args, **kwargs) -> Any:
        """
        Main execution entry point.
        """
        if task == "fetch_data":
            # Wrap the heavy operation with the redundancy checker
            return await self.execute_redundant_task(
                task_name="gallery_data_fetch",
                task_func=self._simulate_heavy_fetch,
                ttl=10, # Short TTL for demo
                use_stale_on_error=True
            )
        return {"error": "Unknown task"}

    async def _simulate_heavy_fetch(self) -> Dict[str, Any]:
        """
        Simulates a network call that might fail or take time.
        """
        # Simulate latency
        await asyncio.sleep(0.5)

        # Simulate random failure (30% chance)
        if random.random() < 0.3:
            raise ConnectionError("Simulated Network Failure in Gallery Backend")

        return {
            "status": "fresh",
            "data": [
                {"id": 1, "name": "Optimized_Agent_1"},
                {"id": 2, "name": "Optimized_Agent_2"}
            ],
            "timestamp": asyncio.get_event_loop().time()
        }

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "OptimizedGalleryAgent",
            "description": "Demonstrates graceful degradation and caching.",
            "skills": ["fetch_data"]
        }
