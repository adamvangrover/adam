import unittest
import sys
import os
import json

# Add repo root to path
sys.path.append(os.getcwd())

from core.agents.specialized.blindspot_agent import BlindspotAgent
from core.engine.consensus_engine import ConsensusEngine
from core.system.agent_orchestrator import AgentOrchestrator, AGENT_CLASSES

class TestVNext(unittest.TestCase):
    def test_blindspot_agent_registration(self):
        """Verify BlindspotAgent is registered in AgentOrchestrator."""
        self.assertIn("BlindspotAgent", AGENT_CLASSES)
        self.assertEqual(AGENT_CLASSES["BlindspotAgent"], BlindspotAgent)

    def test_blindspot_agent_instantiation(self):
        """Verify BlindspotAgent can be instantiated."""
        agent = BlindspotAgent(config={"name": "BlindspotScanner"})
        self.assertIsNotNone(agent)

    def test_consensus_engine(self):
        """Verify ConsensusEngine logic."""
        engine = ConsensusEngine(threshold=0.5)
        signals = [
            {'agent': 'A', 'vote': 'BUY', 'confidence': 0.9, 'weight': 1.0},
            {'agent': 'B', 'vote': 'SELL', 'confidence': 0.1, 'weight': 1.0}
        ]
        # Logic check:
        # A: 1 * 0.9 * 1 = 0.9
        # B: -1 * 0.1 * 1 = -0.1
        # Total: 0.8 / 2 = 0.4.
        # Threshold 0.5. Result should be HOLD (because 0.4 < 0.5).

        result = engine.evaluate(signals)
        self.assertEqual(result['decision'], 'HOLD')
        self.assertEqual(result['score'], 0.4)

    def test_consensus_engine_buy(self):
        engine = ConsensusEngine(threshold=0.5)
        signals = [
            {'agent': 'A', 'vote': 'BUY', 'confidence': 0.9, 'weight': 1.0},
            {'agent': 'B', 'vote': 'BUY', 'confidence': 0.8, 'weight': 1.0}
        ]
        # 0.9 + 0.8 = 1.7 / 2 = 0.85
        result = engine.evaluate(signals)
        self.assertEqual(result['decision'], 'BUY/APPROVE')
        self.assertEqual(result['score'], 0.85)

if __name__ == '__main__':
    unittest.main()
