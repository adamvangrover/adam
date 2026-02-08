
import unittest
import sys
import os
import logging
import json
from datetime import datetime

# Add root to sys.path
sys.path.append(os.getcwd())

class TestAdamVNext(unittest.TestCase):
    """
    Regression tests for ADAM-V-NEXT Protocol components.
    Validates the additive expansion and system-wide enhancements.
    """

    def setUp(self):
        # Configure logging to capture output during tests if needed
        logging.basicConfig(level=logging.INFO)

    def test_consensus_engine(self):
        """Test the ConsensusEngine logic."""
        try:
            from core.engine.consensus_engine import ConsensusEngine
        except ImportError:
            self.fail("Could not import ConsensusEngine")

        engine = ConsensusEngine()
        signals = [
            {'agent': 'Risk', 'vote': 'REJECT', 'confidence': 1.0, 'weight': 10.0, 'reason': 'Too risky'},
            {'agent': 'Tech', 'vote': 'APPROVE', 'confidence': 0.1, 'weight': 1.0, 'reason': 'Weak chart'}
        ]
        result = engine.evaluate(signals)

        self.assertEqual(result['decision'], 'SELL/REJECT', "ConsensusEngine decision logic failed")
        self.assertTrue(result['score'] < -0.6, "ConsensusEngine score logic failed")
        self.assertIn('Risk voted REJECT', result['rationale'], "ConsensusEngine rationale incomplete")

    def test_blindspot_agent(self):
        """Test the BlindspotAgent initialization and basic execution structure."""
        try:
            from core.agents.specialized.blindspot_agent import BlindspotAgent
        except ImportError:
            self.fail("Could not import BlindspotAgent")

        config = {'agent_id': 'Blindspot', 'llm_config': {'model': 'gpt-4'}}
        agent = BlindspotAgent(config)
        self.assertEqual(agent.name, 'Blindspot', "BlindspotAgent initialization failed")

        # We can't easily run async execute in unittest without async support or mocking,
        # but initialization confirms dependencies are met.

    def test_governance_middleware(self):
        """Test the GovernanceMiddleware initialization."""
        try:
            from services.webapp.governance import GovernanceMiddleware
            from flask import Flask
        except ImportError:
            self.fail("Could not import GovernanceMiddleware or Flask")

        app = Flask(__name__)
        gov = GovernanceMiddleware(app)
        self.assertTrue(gov, "GovernanceMiddleware failed to initialize")
        # Check if policy loaded (even if default)
        self.assertIn('global_policy', gov.policy, "Governance policy not loaded")

    def test_narrative_logger(self):
        """Test the NarrativeLogger functionality."""
        try:
            from core.utils.logging_utils import NarrativeLogger
        except ImportError:
            self.fail("Could not import NarrativeLogger")

        logger = NarrativeLogger()
        # NarrativeLogger uses standard logging, so we verify it doesn't crash
        try:
            logger.log_narrative("Test Event", "Analysis", "Decision", "Outcome")
        except Exception as e:
            self.fail(f"NarrativeLogger crashed: {e}")

if __name__ == '__main__':
    unittest.main()
