import os
import sys
import unittest

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.v30_architecture.python_intelligence.agents.code_weaver import CodeWeaverAgent
from core.v30_architecture.python_intelligence.agents.news_bot import NewsBotAgent
from core.v30_architecture.python_intelligence.mcp.server import mcp_server
from core.v30_architecture.python_intelligence.orchestrator.v30_orchestrator import orchestrator


class TestV30Architecture(unittest.TestCase):

    def test_mcp_server_tools(self):
        # Test get_market_data
        res = mcp_server.call_tool("get_market_data", {"symbol": "AAPL"})
        self.assertEqual(res['result']['symbol'], "AAPL")

        # Test sensitive tool (should be auto-approved in test env based on my modification)
        res = mcp_server.call_tool("execute_trade", {"symbol": "AAPL", "side": "BUY", "quantity": 10})
        self.assertEqual(res['result']['status'], "Working")

    def test_orchestrator_decomposition(self):
        res = orchestrator.process_intent("I want to buy some AAPL", "User")
        self.assertEqual(res['status'], "completed")
        trace = res['trace']
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]['step'], "Check Market Data for AAPL")
        self.assertEqual(trace[1]['step'], "Execute Buy Order")

    def test_news_bot(self):
        bot = NewsBotAgent()
        alerts = bot.run_cycle()
        # Should mock finding an alert for Tesla based on my implementation
        found = False
        for item in alerts:
            if item['symbol'] == 'TSLA' and item['sentiment'] < 0:
                found = True
        self.assertTrue(found)

    def test_code_weaver(self):
        # Point to a directory that definitely exists
        weaver = CodeWeaverAgent(repo_path="core/v30_architecture/python_intelligence")
        issues = weaver.scan_for_debt()
        self.assertIsInstance(issues, list)

if __name__ == '__main__':
    unittest.main()
