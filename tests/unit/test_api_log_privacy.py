from core.api.server import app, setup_log_capture
import sys
import os
import unittest
import logging
import json
from unittest.mock import MagicMock, patch

# Adjust path to find core
sys.path.append(os.path.abspath(os.getcwd()))

# Mock dependencies
sys.modules['core.engine.meta_orchestrator'] = MagicMock()
sys.modules['core.system.agent_orchestrator'] = MagicMock()
sys.modules['core.engine.neuro_symbolic_planner'] = MagicMock()
sys.modules['core.engine.states'] = MagicMock()
sys.modules['core.engine.red_team_graph'] = MagicMock()
sys.modules['core.engine.esg_graph'] = MagicMock()
sys.modules['core.engine.regulatory_compliance_graph'] = MagicMock()
sys.modules['core.engine.crisis_simulation_graph'] = MagicMock()
sys.modules['core.engine.deep_dive_graph'] = MagicMock()
sys.modules['core.engine.reflector_graph'] = MagicMock()
sys.modules['core.mcp.registry'] = MagicMock()
sys.modules['core.agents.specialized.management_assessment_agent'] = MagicMock()
sys.modules['core.agents.fundamental_analyst_agent'] = MagicMock()
sys.modules['core.agents.specialized.peer_comparison_agent'] = MagicMock()
sys.modules['core.agents.specialized.snc_rating_agent'] = MagicMock()
sys.modules['core.agents.specialized.covenant_analyst_agent'] = MagicMock()
sys.modules['core.agents.specialized.monte_carlo_risk_agent'] = MagicMock()
sys.modules['core.agents.specialized.quantum_scenario_agent'] = MagicMock()
sys.modules['core.agents.specialized.portfolio_manager_agent'] = MagicMock()
sys.modules['core.schemas.v23_5_schema'] = MagicMock()

# Import app. setup_log_capture exists but does nothing now.


class TestLogLeak(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        setup_log_capture()

    def test_log_leak(self):
        # 1. Generate a sensitive log
        secret = "SUPER_SECRET_KEY_123"
        logging.getLogger('core').info(f"Processing request with key: {secret}")

        # 2. Call the API
        response = self.app.get('/api/state')
        data = json.loads(response.data)

        # 3. Check logs
        logs = data.get('logs')

        # 4. Assert logs are missing or empty
        self.assertTrue(logs is None or len(logs) == 0, "Logs field should be missing or empty")

        # Double check secret is not there in case logs are present
        if logs:
            found = False
            for log in logs:
                if secret in log:
                    found = True
                    break
            self.assertFalse(found, "Secret found in logs!")


if __name__ == '__main__':
    unittest.main()
