import unittest
import asyncio
from core.agents.specialized.credit_risk_controller_agent import CreditRiskControllerAgent
from core.system.v22_async.async_task import AsyncTask

class TestCreditRiskControllerAgent(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_logic(self):
        # Setup
        config = {"model": "mock-model"}
        agent = CreditRiskControllerAgent(agent_id="test_crc", config=config)

        # 1. Test S&P FRP Logic
        # Highly Leveraged Case
        desc, anchor = agent._calculate_sp_frp(leverage=6.5, coverage=2.5)
        self.assertEqual(desc, "Highly Leveraged")
        self.assertEqual(anchor, "B")

        # Modest Case
        desc, anchor = agent._calculate_sp_frp(leverage=1.8, coverage=16.0)
        self.assertEqual(desc, "Modest")
        self.assertEqual(anchor, "BBB")

        # 2. Regulatory Disagreement Logic
        # High Risk: Lev > 6.0 AND Repayment < 50
        risk, drivers = agent._simulate_regulatory_response(leverage=6.5, repayment_capacity=40.0, tdr_flag=False)
        self.assertEqual(risk, "High")
        self.assertIn("Leverage > 6.0x AND Repayment Capacity < 50% (No De-leveraging path)", drivers)

        # Critical Risk: TDR
        risk, drivers = agent._simulate_regulatory_response(leverage=4.0, repayment_capacity=80.0, tdr_flag=True)
        self.assertEqual(risk, "Critical")

        # 3. Conviction Score
        # Match (40) + Tight (5) + Critical Liquidity (0) + Negative Trend (0) = 45
        score, breakdown = agent._calculate_conviction_score(
            internal_rating="B", implied_anchor="B",
            covenant_headroom=10.0, liquidity_percent=5.0, ebitda_growth=-5.0
        )
        self.assertEqual(score, 45)
        self.assertEqual(breakdown['alignment'], "Match (+40)")

    async def test_execute_flow(self):
        config = {"model": "mock-model"}
        agent = CreditRiskControllerAgent(agent_id="test_crc_exec", config=config)

        input_data = {
            'borrower_name': 'Acme Corp',
            'total_debt_to_ebitda': 6.5,
            'ebitda_interest_coverage': 2.5,
            'repayment_capacity_7yr_percent': 40.0,
            'tdr_flag': False,
            'internal_rating': 'BB',
            'covenant_headroom_percent': 10.0,
            'liquidity_percent_of_commit': 5.0,
            'ebitda_growth_yoy': -5.0
        }

        result = await agent.execute(task="Audit", **input_data)

        self.assertEqual(result['status'], "success")
        self.assertEqual(result['metadata']['regulatory_risk'], "High")
        self.assertIn("LLM not available", result['output']['v23_knowledge_graph']['credit_analysis']['generated_report'])

if __name__ == '__main__':
    unittest.main()
