
import unittest
from unittest.mock import MagicMock, patch
from core.data_sources.political_landscape import PoliticalLandscapeLoader
from core.agents.regulatory_compliance_agent import RegulatoryComplianceAgent

class TestPoliticalLandscapeLoader(unittest.TestCase):

    @patch('core.data_sources.political_landscape.requests.get')
    def test_load_landscape_success(self, mock_get):
        # Mock Reuters response
        mock_response_reuters = MagicMock()
        mock_response_reuters.status_code = 200
        mock_response_reuters.text = """
        <html>
            <body>
                <h3>Political Article 1 - A very long headline for testing purposes</h3>
                <h3>Political Article 2 - Another very long headline for testing purposes</h3>
            </body>
        </html>
        """

        # Mock White House response
        mock_response_wh = MagicMock()
        mock_response_wh.status_code = 200
        mock_response_wh.text = """
        <html>
            <body>
                <h2 class="news-item__title">WH Update 1: Official Statement</h2>
            </body>
        </html>
        """

        # Set up side_effect for requests.get to return different responses based on URL
        def side_effect(url, headers=None, timeout=None):
            if "reuters" in url:
                return mock_response_reuters
            elif "whitehouse" in url:
                return mock_response_wh
            return MagicMock(status_code=404)

        mock_get.side_effect = side_effect

        loader = PoliticalLandscapeLoader()
        landscape = loader.load_landscape()

        self.assertIn("US", landscape)
        self.assertEqual(landscape["US"]["president"], "Joe Biden")

        # Check if developments were scraped
        developments = landscape["US"]["recent_developments"]

        # We expect formatted strings like "[Reuters] ..."
        self.assertTrue(any("Political Article 1" in d for d in developments))
        self.assertTrue(any("WH Update 1" in d for d in developments))

    @patch('core.data_sources.political_landscape.requests.get')
    def test_load_landscape_failure(self, mock_get):
        # Simulate network error
        mock_get.side_effect = Exception("Network Down")

        loader = PoliticalLandscapeLoader()
        landscape = loader.load_landscape()

        self.assertIn("US", landscape)
        # Should fall back to default data
        self.assertIn("Ongoing implementation of climate policies.", landscape["US"]["recent_developments"][0])

class TestRegulatoryComplianceAgentIntegration(unittest.TestCase):
    @patch('core.agents.regulatory_compliance_agent.GraphDatabase')
    @patch('core.agents.regulatory_compliance_agent.LLMPlugin')
    @patch('core.agents.regulatory_compliance_agent.nltk')
    def test_agent_initialization(self, mock_nltk, mock_llm, mock_graph):
        # Mock NLTK downloads to avoid actual download during test
        mock_nltk.download.return_value = True

        config = {"llm_config": {}}
        agent = RegulatoryComplianceAgent(config)

        # Check if political landscape is loaded
        self.assertIn("US", agent.political_landscape)
        self.assertIsNotNone(agent.political_landscape["US"].get("president"))

if __name__ == '__main__':
    unittest.main()
