
from core.engine.valuation_utils import calculate_dcf, calculate_multiples, get_price_targets
import unittest
import sys
import os

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDCFValuation(unittest.TestCase):

    def setUp(self):
        self.financials = {
            "fcf": 1000,
            "growth_rate": 0.05,
            "beta": 1.2,
            "shares_outstanding": 500,
            "debt_equity_ratio": 0.5,
            "net_debt": 2000
        }
        self.risk_free_rate = 0.04

    def test_calculate_dcf_basic(self):
        """Test basic DCF calculation with default parameters."""
        result = calculate_dcf(self.financials, self.risk_free_rate)

        self.assertIn("wacc", result)
        self.assertIn("intrinsic_share_price", result)
        self.assertIsInstance(result["intrinsic_share_price"], float)
        self.assertGreater(result["intrinsic_value"], 0)

        # Verify WACC calculation roughly
        # Cost of Equity = 0.04 + 1.2 * 0.05 = 0.10
        # Cost of Debt (after tax) = 0.06 * (1 - 0.21) = 0.0474
        # D/E = 0.5 -> W_e = 1/1.5 = 0.666, W_d = 0.5/1.5 = 0.333
        # WACC = 0.10 * 0.666 + 0.0474 * 0.333 = 0.0666 + 0.0158 = ~0.0824

        self.assertAlmostEqual(result["wacc"], 0.0825, places=3)

    def test_calculate_dcf_scenario_injection(self):
        """Test DCF with different risk free rate (Scenario Injection)."""
        # Scenario: Rates hit 6%
        high_rates = 0.06
        result_high_rates = calculate_dcf(self.financials, risk_free_rate=high_rates)

        # Scenario: Rates drop to 2%
        low_rates = 0.02
        result_low_rates = calculate_dcf(self.financials, risk_free_rate=low_rates)

        # Higher rates should increase WACC and decrease Intrinsic Value
        self.assertGreater(result_high_rates["wacc"], result_low_rates["wacc"])
        self.assertLess(result_high_rates["intrinsic_share_price"], result_low_rates["intrinsic_share_price"])

    def test_calculate_dcf_full_scenario_override(self):
        """Test comprehensive scenario injection (recession)."""
        # Scenario: Recession (High Risk Premium, Low Growth)
        recession_scenario = {
            "market_risk_premium": 0.08,  # Default 0.05
            "growth_rate": 0.01,         # Default 0.05
            "risk_free_rate": 0.02       # Rate cut
        }

        result_base = calculate_dcf(self.financials, self.risk_free_rate)
        result_recession = calculate_dcf(self.financials, scenario=recession_scenario)

        # Recession should crush the valuation despite lower rates
        self.assertLess(result_recession["intrinsic_share_price"], result_base["intrinsic_share_price"])

    def test_calculate_multiples(self):
        financials = {"enterprise_value": 50000, "ebitda": 2500}  # EV/EBITDA = 20x
        peers = [{"ev_ebitda": 15.0}, {"ev_ebitda": 25.0}]  # Median = 20x

        result = calculate_multiples(financials, peers)
        self.assertEqual(result["current_ev_ebitda"], 20.0)
        self.assertEqual(result["verdict"], "Fairly Valued")

        # Test Overvalued
        financials_expensive = {"enterprise_value": 75000, "ebitda": 2500}  # 30x
        result_exp = calculate_multiples(financials_expensive, peers)
        self.assertEqual(result_exp["verdict"], "Overvalued")


if __name__ == "__main__":
    unittest.main()
