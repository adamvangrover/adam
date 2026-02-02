import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch
import asyncio

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.engine.consensus_engine import ConsensusEngine
    from core.agents.specialized.blindspot_agent import BlindspotAgent
    from services.webapp.governance import GovernanceMiddleware
    from core.utils.logging_utils import NarrativeLogger
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

class TestAdamVNext(unittest.TestCase):
    def test_consensus_engine(self):
        """Task 2.1: Consensus Engine"""
        print("\nTesting Consensus Engine...")
        engine = ConsensusEngine(threshold=0.5)
        signals = [
            {'agent': 'A1', 'vote': 'BUY', 'confidence': 0.8, 'weight': 1.0},
            {'agent': 'A2', 'vote': 'SELL', 'confidence': 0.2, 'weight': 1.0}
        ]
        result = engine.evaluate(signals)
        # Score contrib:
        # A1: 1.0 * 0.8 * 1.0 = 0.8
        # A2: -1.0 * 0.2 * 1.0 = -0.2
        # Total: 0.6. Total Weight: 2.0. Final: 0.3
        self.assertAlmostEqual(result['score'], 0.3)
        self.assertEqual(result['decision'], "HOLD") # 0.3 < 0.5
        print("Consensus Engine Test Passed.")

    def test_blindspot_agent(self):
        """Task 2.2: Blindspot Agent"""
        print("\nTesting Blindspot Agent...")
        agent = BlindspotAgent({'name': 'TestBlindspot', 'llm_config': {}})

        # Mocking Neo4j driver to avoid actual connection attempts or failures
        with patch('core.agents.specialized.blindspot_agent.get_neo4j_driver', return_value=None):
            # Also need to mock live_engine if it's imported inside execute
            with patch.dict(sys.modules, {'core.engine.live_mock_engine': MagicMock()}):
                 # Mock the live engine's pulse
                 mock_live = MagicMock()
                 mock_live.get_market_pulse.return_value = {'sectors': {}, 'indices': {}}

                 with patch('core.engine.live_mock_engine.live_engine', mock_live):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(agent.execute())

                    self.assertIn("status", result)
                    self.assertEqual(result["status"], "SCAN_COMPLETE")
                    # Should find simulated anomalies if Neo4j is None (NetworkX fallback)
                    self.assertTrue(len(result['findings']) > 0)
                    print(f"Blindspot Agent found {len(result['findings'])} anomalies (Simulated).")

    def test_governance_middleware(self):
        """Task 4.1: Governance Middleware"""
        print("\nTesting Governance Middleware...")
        app = MagicMock()
        # Mock open to avoid file not found if config missing, but we expect it to exist
        if not os.path.exists('config/governance_policy.yaml'):
             print("Warning: config/governance_policy.yaml not found. Test might fail.")

        mw = GovernanceMiddleware(app, policy_path='config/governance_policy.yaml')
        self.assertIn('restricted_endpoints', mw.policy)
        print("Governance Middleware Policy Loaded.")

    def test_narrative_logger(self):
        """Task 4.2: Structured Logging"""
        print("\nTesting Narrative Logger...")
        logger = NarrativeLogger("TestLogger")
        # Just check it doesn't crash
        logger.log_narrative("Event", "Analysis", "Decision", "Outcome")
        print("Narrative Logger executed.")

if __name__ == '__main__':
    unittest.main()
