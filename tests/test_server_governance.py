import sys
import os
import unittest
import json
from unittest.mock import MagicMock

# Define exception classes for mocking
class GovernanceError(Exception): pass
class ApprovalRequired(GovernanceError): pass

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Mock dependencies
mock_gov_module = MagicMock()
mock_gov_module.GovernanceError = GovernanceError
mock_gov_module.ApprovalRequired = ApprovalRequired

sys.modules['core.security.governance'] = mock_gov_module
mock_sandbox_module = MagicMock()
sys.modules['core.security.sandbox'] = mock_sandbox_module

# Mock other dependencies to avoid import errors
sys.modules['core.vertical_risk_agent.generative_risk'] = MagicMock()
sys.modules['core.v22_quantum_pipeline.qmc_engine'] = MagicMock()
sys.modules['core.engine.meta_orchestrator'] = MagicMock()
sys.modules['core.credit_sentinel.agents.ratio_calculator'] = MagicMock()
sys.modules['core.credit_sentinel.models.distress_classifier'] = MagicMock()
sys.modules['core.credit_sentinel.agents.risk_analyst'] = MagicMock()
sys.modules['core.governance.constitution'] = MagicMock()
sys.modules['pandas'] = MagicMock()

# Import server
sys.path.insert(0, os.path.join(PROJECT_ROOT, "server"))

# Force reload/import of server to pick up the mocked exceptions
if 'server' in sys.modules:
    del sys.modules['server']
# Also need to clear server_module if it exists
if 'server_module' in sys.modules:
    del sys.modules['server_module']

try:
    import server
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("server_module", os.path.join(PROJECT_ROOT, "server", "server.py"))
    server = importlib.util.module_from_spec(spec)
    sys.modules["server_module"] = server
    spec.loader.exec_module(server)

class TestServerGovernance(unittest.TestCase):

    def setUp(self):
        self.mock_gov = mock_gov_module.GovernanceEnforcer
        self.mock_sandbox = mock_sandbox_module.SecureSandbox

        self.mock_gov.reset_mock()
        self.mock_sandbox.reset_mock()
        self.mock_gov.validate.side_effect = None

        # Ensure dependencies are "available"
        server.GOVERNANCE_AVAILABLE = True

    def test_execute_python_sandbox_calls_governance(self):
        """Test that execute_python_sandbox calls GovernanceEnforcer.validate"""

        # Setup
        self.mock_sandbox.execute.return_value = {"status": "success", "output": "test"}
        code = "print('test')"

        # Execute
        result = server.execute_python_sandbox(code)

        # Verify
        self.mock_gov.validate.assert_called()
        self.mock_sandbox.execute.assert_called_with(code)

    def test_execute_python_sandbox_blocks_risky_code(self):
        """Test that execute_python_sandbox handles GovernanceError"""

        # Setup
        # Use the exception class defined above which is also in the mock module
        self.mock_gov.validate.side_effect = GovernanceError("Risky code detected")
        code = "import os"

        # Execute
        result_json = server.execute_python_sandbox(code)
        result = json.loads(result_json)

        # Verify
        self.assertEqual(result.get("status"), "blocked")
        self.assertIn("Risky code detected", result.get("error", ""))
        self.mock_sandbox.execute.assert_not_called()

if __name__ == '__main__':
    unittest.main()
