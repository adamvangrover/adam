import unittest
from unittest.mock import patch, MagicMock
import datetime
import sys
import os

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock missing modules to allow import
sys.modules['utils'] = MagicMock()
sys.modules['utils.data_validation'] = MagicMock()
sys.modules['utils.visualization_tools'] = MagicMock()

# Try to import, if it fails due to other missing deps, we might need more mocks
try:
    from core.agents.event_driven_risk_agent import EventDrivenRiskAgent
except ImportError as e:
    # If import fails, we can't test it easily.
    # But since I'm fixing a specific file, maybe I can just test that file?
    # No, I need to instantiate the class.
    raise e

class TestEventDrivenRiskAgentSecurity(unittest.TestCase):
    def setUp(self):
        # We need to patch os.getenv if it's used in __init__
        with patch('os.getenv', return_value='fake_key'):
            self.agent = EventDrivenRiskAgent()

    @patch('core.agents.event_driven_risk_agent.requests.get')
    def test_fetch_events_has_timeout(self, mock_get):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "articles": []}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Call the method
        keywords = ["test"]
        from_date = datetime.date(2024, 1, 1)
        to_date = datetime.date(2024, 1, 2)

        # We also need to mock validate_event_data since it's imported
        with patch('core.agents.event_driven_risk_agent.validate_event_data', return_value=True):
             self.agent.fetch_events(keywords, from_date, to_date)

        # Verify calls
        self.assertTrue(mock_get.called, "requests.get should have been called")

        args, kwargs = mock_get.call_args
        self.assertIn('timeout', kwargs, "requests.get should be called with a timeout")
        self.assertGreater(kwargs['timeout'], 0, "Timeout should be greater than 0")

if __name__ == '__main__':
    unittest.main()
