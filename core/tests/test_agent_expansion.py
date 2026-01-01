import unittest
import asyncio
from core.agents.specialized.strategic_snc_agent import StrategicSNCAgent
from core.agents.specialized.regulatory_snc_agent import RegulatorySNCAgent
from core.agents.specialized.technical_covenant_agent import TechnicalCovenantAgent
from core.agents.specialized.financial_covenant_agent import CovenantAnalystAgent

class TestAgentExpansion(unittest.TestCase):

    def setUp(self):
        self.config = {"name": "TestAgent", "llm_config": {}}

    def test_strategic_snc_agent_load(self):
        agent = StrategicSNCAgent(self.config)
        self.assertEqual(agent.persona, "Bank Risk Officer")
        self.assertTrue(hasattr(agent, 'consensus_engine'))

    def test_regulatory_snc_agent_load(self):
        agent = RegulatorySNCAgent(self.config)
        self.assertEqual(agent.persona, "FDIC Examiner")

    def test_technical_covenant_agent_load(self):
        agent = TechnicalCovenantAgent(self.config)
        self.assertEqual(agent.persona, "Legal Associate")

    def test_financial_covenant_agent_load(self):
        # Note: In the file system, this is still named CovenantAnalystAgent inside financial_covenant_agent.py
        # because I renamed the file but not the class to maintain some compatibility or maybe I should have?
        # Let's check the import above.
        agent = CovenantAnalystAgent(self.config)
        self.assertEqual(agent.persona, "Credit Lawyer")

    def test_dual_agent_execution_flow(self):
        # Verify both SNC agents can run on same data
        fin = {"ebitda": 100, "total_debt": 300, "interest_expense": 20}
        cap = [{"name": "TLB", "amount": 300}]
        ev = 500

        reg_agent = RegulatorySNCAgent(self.config)
        strat_agent = StrategicSNCAgent(self.config)

        reg_result = asyncio.run(reg_agent.execute(fin, cap, ev))
        strat_result = asyncio.run(strat_agent.execute(fin, cap, ev))

        # Lev = 3.0x. Both should pass.
        self.assertEqual(reg_result.overall_borrower_rating, "Pass")
        self.assertEqual(strat_result.overall_borrower_rating, "Pass")

        # Reg conviction should be 1.0, Strat might be different
        self.assertEqual(reg_result.conviction_score, 1.0)
        self.assertNotEqual(strat_result.conviction_score, 0.0)

if __name__ == '__main__':
    unittest.main()
