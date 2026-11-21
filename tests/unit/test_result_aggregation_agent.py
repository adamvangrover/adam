# tests/test_result_aggregation_agent.py

import unittest
from core.agents.result_aggregation_agent import ResultAggregationAgent
from core.system.error_handler import AdamError


import unittest
import asyncio

class TestResultAggregationAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.agent = ResultAggregationAgent({'name': 'ResultAggregator'})

    async def test_execute_empty_list(self):
        result = await self.agent.execute([])
        self.assertEqual(result, "No results to aggregate.")

    async def test_execute_single_result(self):
        result = await self.agent.execute(["Result 1"])
        self.assertEqual(result, "Result 1")

    async def test_execute_multiple_results(self):
        result = await self.agent.execute(["Result 1", "Result 2", "Result 3"])
        self.assertEqual(result, "Result 1\n\nResult 2\n\nResult 3")

    async def test_execute_with_error(self):
        # Simulate an agent returning an error object
        error = AdamError(999, "Test Error")  # Use a dummy error code
        result = await self.agent.execute(["Result 1", error, "Result 3"])
        self.assertEqual(result, "Result 1\n\nError Code 999: Test Error\n\nResult 3")

    async def test_execute_mixed_types(self):
        result = await self.agent.execute(["Result 1", 123, {"key": "value"}])
        self.assertEqual(result, "Result 1\n\n123\n\n{'key': 'value'}")


if __name__ == '__main__':
    unittest.main()
