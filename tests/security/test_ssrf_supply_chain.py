import sys
from unittest.mock import MagicMock

# Mock missing dependencies
sys.modules['folium'] = MagicMock()
sys.modules['geopy'] = MagicMock()
sys.modules['geopy.geocoders'] = MagicMock()

import unittest
from unittest.mock import patch
from core.agents.supply_chain_risk_agent import SupplyChainRiskAgent

class TestSupplyChainRiskAgentSSRF(unittest.TestCase):
    @patch('requests.get')
    def test_ssrf_mitigation(self, mock_get):
        # Setup - Agent with malicious URLs
        malicious_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost:5000/admin",
            "file:///etc/passwd",
            "ftp://example.com/file"
        ]
        agent = SupplyChainRiskAgent(
            news_api_key="fake_key",
            supplier_data=[],
            transportation_routes=[],
            geopolitical_data=[],
            web_scraping_urls=malicious_urls
        )

        # Execute
        agent.fetch_web_scraped_data()

        # Verify - The agent should NOT request any of these URLs
        mock_get.assert_not_called()

    @patch('requests.get')
    def test_valid_url(self, mock_get):
        # Setup - Agent with valid URL
        valid_url = "https://www.example.com"
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"<html><p>Safe content</p></html>"

        agent = SupplyChainRiskAgent(
            news_api_key="fake_key",
            supplier_data=[],
            transportation_routes=[],
            geopolitical_data=[],
            web_scraping_urls=[valid_url]
        )

        # Execute
        data = agent.fetch_web_scraped_data()

        # Verify
        mock_get.assert_called_with(valid_url, timeout=10)
        self.assertIn("Safe content", data)

if __name__ == '__main__':
    unittest.main()
