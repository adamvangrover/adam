import unittest
import numpy as np
import asyncio
from core.agents.quantitative_risk_agent import QuantitativeRiskAgent

class TestQuantitativeRiskAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QuantitativeRiskAgent(config={"confidence_level": 0.95})
        np.random.seed(42)
        # Generate some normal returns
        self.returns = np.random.normal(0.001, 0.02, 1000)

    def test_var_calculation(self):
        result = asyncio.run(self.agent.execute(returns=self.returns))
        self.assertEqual(result['status'], 'success')
        self.assertIn('VaR', result['metrics'])
        # VaR should be negative (loss)
        self.assertLess(result['metrics']['VaR'], 0)

    def test_legacy_financial_modeling_agent_init(self):
        # Verify backward compatibility fix
        from core.agents.financial_modeling_agent import FinancialModelingAgent
        # Old style init
        agent = FinancialModelingAgent(initial_cash_flow=500, discount_rate=0.05)
        self.assertEqual(agent.initial_cash_flow, 500)
        self.assertEqual(agent.discount_rate, 0.05)

    def test_legacy_portfolio_agent_class(self):
        # Verify backward compatibility fix
        from core.agents.portfolio_optimization_agent import AIPoweredPortfolioOptimizationAgent
        agent = AIPoweredPortfolioOptimizationAgent(config={})
        self.assertTrue(isinstance(agent, AIPoweredPortfolioOptimizationAgent))

if __name__ == '__main__':
    unittest.main()
