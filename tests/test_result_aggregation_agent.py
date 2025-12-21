# tests/test_result_aggregation_agent.py

import unittest

from core.agents.result_aggregation_agent import ResultAggregationAgent
from core.system.error_handler import AdamError


class TestResultAggregationAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ResultAggregationAgent()

    def test_execute_empty_list(self):
        result = self.agent.execute([])
        self.assertEqual(result, "")

    def test_execute_single_result(self):
        result = self.agent.execute(["Result 1"])
        self.assertEqual(result, "Result 1")

    def test_execute_multiple_results(self):
        result = self.agent.execute(["Result 1", "Result 2", "Result 3"])
        self.assertEqual(result, "Result 1\nResult 2\nResult 3")

    def test_execute_with_error(self):
        # Simulate an agent returning an error object
        error = AdamError(999, "Test Error")  # Use a dummy error code
        result = self.agent.execute(["Result 1", error, "Result 3"])
        self.assertEqual(result, "Result 1\nError Code 999: Test Error\nResult 3")

    def test_execute_mixed_types(self):
        result = self.agent.execute(["Result 1", 123, {"key": "value"}])
        self.assertEqual(result, "Result 1\n123\n{'key': 'value'}")

if __name__ == '__main__':
    unittest.main()
