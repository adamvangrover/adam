import unittest
from unittest.mock import patch, Mock
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent

class TestGeopoliticalRiskAgent(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of GeopoliticalRiskAgent."""
        config = {'data_sources': {'news_api': Mock(), 'political_database': Mock()}}
        self.agent = GeopoliticalRiskAgent(config)

    @patch('core.agents.geopolitical_risk_agent.send_message')
    def test_assess_geopolitical_risks(self, mock_send_message):
        """Test assessing geopolitical risks."""
        # ... (mock data sources and their responses)

        risk_assessments = self.agent.assess_geopolitical_risks()
        self.assertIsInstance(risk_assessments, dict)
        self.assertIn('political_risk_index', risk_assessments)
        self.assertIn('key_risks', risk_assessments)
        mock_send_message.assert_called_once_with(
            {'agent': 'geopolitical_risk_agent', 'risk_assessments': risk_assessments}
        )
