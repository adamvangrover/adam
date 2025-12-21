import unittest

from core.financial_suite.context_manager import ContextManager
from core.financial_suite.schemas.workstream_context import WorkstreamContext


class TestFinancialSuite(unittest.TestCase):
    def setUp(self):
        self.fixture_path = "tests/fixtures/sample_context.json"

    def test_load_context(self):
        manager = ContextManager(context_path=self.fixture_path)
        self.assertIsInstance(manager.context, WorkstreamContext)
        self.assertEqual(manager.context.config.mode, "VC_SPONSOR")

    def test_run_workstream(self):
        manager = ContextManager(context_path=self.fixture_path)
        results = manager.run_workstream()

        # Solver
        self.assertIn("solver", results)
        val = results["solver"]["valuation"]
        self.assertGreater(val["enterprise_value"], 0)
        print(f"EV: {val['enterprise_value']}")

        # Metrics
        metrics = results["solver"]["metrics"]
        self.assertIn("wacc", metrics)
        print(f"WACC: {metrics['wacc']}")
        self.assertIn("rating", metrics)
        print(f"Rating: {metrics['rating']}")

        # Waterfall
        self.assertIn("waterfall", results)
        waterfall = results["waterfall"]
        print(f"Waterfall: {waterfall}")

        # Report
        self.assertIn("report", results)
        self.assertTrue(results["report"].startswith("# ADAM Financial Workstream Report"))

    def test_sensitivity_generation(self):
        manager = ContextManager(context_path=self.fixture_path)
        manager.run_workstream() # Need to run to set up report? No, report generation runs internally.

        report = manager.results["report"]
        self.assertIn("| Margin \ SOFR |", report)
        self.assertIn("| Rev Contraction \ Volatility |", report)

if __name__ == "__main__":
    unittest.main()
