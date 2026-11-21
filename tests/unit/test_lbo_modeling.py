import unittest
import numpy as np
from core.agents.financial_modeling_agent import FinancialModelingAgent
from core.financial_data.modeling_schema import LBOAssumptions

class TestLBOModeling(unittest.TestCase):
    def setUp(self):
        self.agent = FinancialModelingAgent()
        self.assumptions = LBOAssumptions(
            entry_multiple=10.0,
            exit_multiple=10.0,
            initial_ebitda=100.0,
            debt_amount=500.0,
            interest_rate=0.08,
            equity_contribution=500.0,
            holding_period=5
        )
        self.agent.lbo_assumptions = self.assumptions

    def test_lbo_calculation(self):
        result = self.agent.calculate_lbo()

        # Check basic properties
        self.assertIsNotNone(result)
        self.assertGreater(result.irr, 0.0)
        self.assertGreater(result.mom_multiple, 1.0)
        self.assertEqual(len(result.cash_flows), 6) # Year 0 + 5 years

        # Check debt paydown logic
        self.assertLess(result.final_debt, 500.0) # Debt should decrease

    def test_lbo_negative_equity(self):
        # Scenario where debt is too high
        self.agent.lbo_assumptions.debt_amount = 2000.0
        self.agent.lbo_assumptions.equity_contribution = 100.0

        result = self.agent.calculate_lbo()

        # Should still run, but IRR/MoM might be weird or handled
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
