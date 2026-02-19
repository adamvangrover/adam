# Verified for Adam v25.5
# Reviewed by Jules
# Protocol Verified: ADAM-V-NEXT (Updated)
import os
import sys
import unittest
import logging
from typing import Dict, Any

# Ensure core modules can be imported
sys.path.append(os.getcwd())

class TestAdamVNextProtocol(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ProtocolVerifier")

    def test_consensus_engine(self):
        """Task 2.1: Verify Consensus Engine"""
        try:
            from core.engine.consensus_engine import ConsensusEngine
            engine = ConsensusEngine()
            signals = [
                {'agent': 'Risk', 'vote': 'REJECT', 'confidence': 0.9, 'weight': 2.0, 'reason': 'Test Reject'},
                {'agent': 'Tech', 'vote': 'APPROVE', 'confidence': 0.8, 'weight': 1.0, 'reason': 'Test Approve'}
            ]
            result = engine.evaluate(signals)
            self.assertIn('decision', result)
            self.assertIn('score', result)
            self.assertIn('rationale', result)
            self.logger.info(f"Consensus Engine Verified: {result}")
        except ImportError as e:
            self.fail(f"Could not import ConsensusEngine: {e}")
        except Exception as e:
            self.fail(f"ConsensusEngine failed: {e}")

    def test_blindspot_agent(self):
        """Task 2.2: Verify Blindspot Agent"""
        try:
            from core.agents.specialized.blindspot_agent import BlindspotAgent
            config = {'name': 'BlindspotScanner', 'model': 'gpt-4'}
            agent = BlindspotAgent(config)
            self.assertIsNotNone(agent)
            self.logger.info("Blindspot Agent Instantiated successfully.")
        except ImportError as e:
            self.fail(f"Could not import BlindspotAgent: {e}")
        except Exception as e:
            self.fail(f"BlindspotAgent instantiation failed: {e}")

    def test_governance_middleware(self):
        """Task 4.1: Verify Governance Middleware"""
        try:
            from services.webapp.governance import GovernanceMiddleware
            middleware = GovernanceMiddleware() # No app needed for basic instantiation test
            self.assertIsNotNone(middleware)
            self.logger.info("Governance Middleware Instantiated successfully.")
        except ImportError as e:
            self.fail(f"Could not import GovernanceMiddleware: {e}")
        except Exception as e:
            self.fail(f"GovernanceMiddleware instantiation failed: {e}")

    def test_narrative_logger(self):
        """Task 4.2: Verify Narrative Logger"""
        try:
            from core.utils.logging_utils import NarrativeLogger
            logger = NarrativeLogger()
            self.assertIsNotNone(logger)
            self.logger.info("Narrative Logger Instantiated successfully.")
        except ImportError as e:
            self.fail(f"Could not import NarrativeLogger: {e}")
        except Exception as e:
            self.fail(f"NarrativeLogger instantiation failed: {e}")

    def test_frontend_files_exist(self):
        """Task 1.1 & 1.2 & 3.1: Verify Frontend & Showcase Files"""
        files_to_check = [
            'services/webapp/client/src/pages/Synthesizer.tsx',
            'services/webapp/client/src/components/AgentIntercom.tsx',
            'showcase/war_room_v2.html'
        ]
        for f in files_to_check:
            path = os.path.join(os.getcwd(), f)
            self.assertTrue(os.path.exists(path), f"File missing: {f}")
            self.logger.info(f"File verified: {f}")

if __name__ == '__main__':
    unittest.main()
