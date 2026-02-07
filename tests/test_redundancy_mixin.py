import asyncio
import unittest
import time
from core.agents.mixins.redundancy_optimization_mixin import RedundancyOptimizationMixin

class TestRedundancyMixin(unittest.TestCase):
    def test_caching(self):
        mixin = RedundancyOptimizationMixin()

        # Counter to track executions
        execution_count = 0

        async def mock_task(x):
            nonlocal execution_count
            execution_count += 1
            return x * 2

        async def run_test():
            # First call
            res1 = await mixin.execute_redundant_task("task1", mock_task, 10, ttl=5)
            self.assertEqual(res1, 20)
            self.assertEqual(execution_count, 1)

            # Second call (Should hit cache)
            res2 = await mixin.execute_redundant_task("task1", mock_task, 10, ttl=5)
            self.assertEqual(res2, 20)
            self.assertEqual(execution_count, 1) # Count should NOT increment

            # Third call with different args (Should execute)
            res3 = await mixin.execute_redundant_task("task1", mock_task, 20, ttl=5)
            self.assertEqual(res3, 40)
            self.assertEqual(execution_count, 2)

        asyncio.run(run_test())

    def test_fallback(self):
        mixin = RedundancyOptimizationMixin()
        execution_count = 0

        async def flaking_task():
            nonlocal execution_count
            execution_count += 1
            if execution_count == 2:
                raise ValueError("Simulated Failure")
            return "Success"

        async def run_test():
            # 1. Success
            res1 = await mixin.execute_redundant_task("flake", flaking_task, ttl=0.1, use_stale_on_error=True)
            self.assertEqual(res1, "Success")

            # Sleep to expire TTL
            time.sleep(0.2)

            # 2. Failure (Should fallback to stale)
            res2 = await mixin.execute_redundant_task("flake", flaking_task, ttl=0.1, use_stale_on_error=True)
            self.assertEqual(res2, "Success")

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
