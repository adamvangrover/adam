
import unittest
import numpy as np
import asyncio
from core.agents.financial_modeling_agent import FinancialModelingAgent

class TestFinancialModelingAgent(unittest.TestCase):
    def setUp(self):
        self.config = {
            'forecast_years': 5,
            'industry_multiples': {'EBITDA': 10.0, 'Revenue': 2.0},
            'terminal_valuation_method': 'Gordon Growth',
            'initial_cash_flow': 100,
            'discount_rate': 0.10,
            'growth_rate': 0.05,
            'terminal_growth_rate': 0.02
        }
        self.agent = FinancialModelingAgent(config=self.config)

    def test_initialization(self):
        self.assertEqual(self.agent.initial_cash_flow, 100)
        self.assertEqual(self.agent.discount_rate, 0.10)
        self.assertEqual(self.agent.growth_rate, 0.05)
        # Check new schema-based attr
        self.assertEqual(self.agent.assumptions.initial_cash_flow, 100)

    def test_generate_cash_flows(self):
        cash_flows = self.agent.generate_cash_flows()
        self.assertEqual(len(cash_flows), 5)
        # Year 1 = 100 * 1.05 = 105
        self.assertAlmostEqual(cash_flows[0], 105.0)
        # Year 2 = 105 * 1.05 = 110.25
        self.assertAlmostEqual(cash_flows[1], 110.25)

    def test_calculate_discounted_cash_flows(self):
        self.agent.generate_cash_flows()
        dcfs = self.agent.calculate_discounted_cash_flows()
        # Year 1 DCF = 105 / 1.1 = 95.4545
        self.assertAlmostEqual(dcfs[0], 95.4545, places=4)

    def test_calculate_terminal_value_gordon(self):
        self.agent.generate_cash_flows()
        tv = self.agent.calculate_terminal_value()

        last_cf = self.agent.cash_flows[-1] # Year 5
        # TV = last_cf * (1+g_term) / (r - g_term)
        expected_tv = last_cf * 1.02 / (0.10 - 0.02)
        self.assertAlmostEqual(tv, expected_tv)

    def test_calculate_npv(self):
        self.agent.generate_cash_flows()
        npv = self.agent.calculate_npv()

        # Verify it's a positive number and greater than initial cash flow
        self.assertGreater(npv, 100)

    def test_sensitivity_analysis(self):
        self.agent.generate_cash_flows()
        results = self.agent.perform_sensitivity_analysis([0.04, 0.05, 0.06], variable='growth_rate')
        self.assertEqual(len(results), 3)
        # Higher growth should mean higher NPV
        self.assertGreater(results[0.06], results[0.04])

    def test_monte_carlo_simulation(self):
        # Ensure base state is ready
        self.agent.generate_cash_flows()

        results = self.agent.run_monte_carlo_simulation(num_simulations=100)
        self.assertIn('mean_npv', results)
        self.assertIn('std_dev', results)
        self.assertEqual(results['num_simulations'], 100)
        # Mean should be relatively close to deterministic NPV
        det_npv = self.agent.calculate_npv()
        self.assertTrue(abs(results['mean_npv'] - det_npv) < det_npv * 0.2) # Allow 20% variance

    def test_financial_ratios(self):
        mock_data = {
            'revenue': [1000],
            'ebitda': [200],
            'ebit': [150],
            'interest_expense': [50],
            'total_debt': [500],
            'cash_and_equivalents': [100],
            'total_assets': [2000],
            'current_assets': [600],
            'current_liabilities': [300]
        }
        ratios = self.agent.calculate_financial_ratios(mock_data)

        self.assertAlmostEqual(ratios['EBITDA_Margin'], 0.20)
        self.assertAlmostEqual(ratios['Net_Debt_to_EBITDA'], (500-100)/200) # 2.0
        self.assertAlmostEqual(ratios['Interest_Coverage'], 150/50) # 3.0
        self.assertAlmostEqual(ratios['Current_Ratio'], 600/300) # 2.0

    def test_execute_dispatch(self):
        # Test task="ratios"
        res = asyncio.run(self.agent.execute(task="ratios", company_id="test_co"))
        # Since we didn't provide data and fetch returns mock data, it should calculate based on mock
        self.assertEqual(res['status'], 'success')
        self.assertIn('ratios', res)

        # Test task="monte_carlo"
        res_mc = asyncio.run(self.agent.execute(task="monte_carlo", num_simulations=10))
        self.assertEqual(res_mc['status'], 'success')
        self.assertIn('results', res_mc)

        # Test default DCF with new schema structure
        res_dcf = asyncio.run(self.agent.execute(task="dcf", company_id="test_co"))
        self.assertEqual(res_dcf['status'], 'success')
        self.assertIn('valuation_model', res_dcf)
        self.assertIn('glossary', res_dcf)

    def test_sentiment_adjustment(self):
        original_growth = self.agent.growth_rate
        original_discount = self.agent.discount_rate

        # Apply Bullish Sentiment (+0.8)
        # Growth should increase, Discount should decrease
        asyncio.run(self.agent.execute(task="dcf", sentiment_score=0.8))

        self.assertGreater(self.agent.assumptions.growth_rate, original_growth)
        self.assertLess(self.agent.assumptions.discount_rate, original_discount)
        self.assertGreater(self.agent.assumptions.sentiment_adjustment_factor, 1.0)

if __name__ == '__main__':
    unittest.main()
