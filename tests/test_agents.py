# tests/test_agents.py

import unittest
from unittest.mock import patch, Mock
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
#... (import other agents)

class TestMarketSentimentAgent(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of MarketSentimentAgent."""
        config = {
            'data_sources': {'financial_news_api': Mock()},  # Mock the data source
            'sentiment_threshold': 0.6
        }
        self.agent = MarketSentimentAgent(config)

    @patch('core.agents.market_sentiment_agent.send_message')
    def test_analyze_sentiment(self, mock_send_message):
        """Test analyzing market sentiment."""
        # Mock the data source's response
        self.agent.data_sources['financial_news_api'].get_headlines.return_value = (
            ["Positive headline", "Negative headline"], "neutral"
        )

        sentiment = self.agent.analyze_sentiment()
        self.assertEqual(sentiment, "neutral")  # Check if sentiment is calculated correctly
        mock_send_message.assert_called_once_with(
            {'agent': 'market_sentiment_agent', 'sentiment': 'neutral'}
        )  # Check if message is sent

class TestMacroeconomicAnalysisAgent(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of MacroeconomicAnalysisAgent."""
        config = {
            'data_sources': {'government_stats_api': Mock()},
            'indicators': ['GDP', 'inflation']
        }
        self.agent = MacroeconomicAnalysisAgent(config)

    @patch('core.agents.macroeconomic_analysis_agent.send_message')
    def test_analyze_macroeconomic_data(self, mock_send_message):
        """Test analyzing macroeconomic data."""
        # Mock the data source's responses
        self.agent.data_sources['government_stats_api'].get_gdp.return_value =
        self.agent.data_sources['government_stats_api'].get_inflation.return_value =

        insights = self.agent.analyze_macroeconomic_data()
        self.assertIsInstance(insights, dict)  # Check if insights is a dictionary
        self.assertIn('GDP_growth_trend', insights)
        self.assertIn('inflation_outlook', insights)
        mock_send_message.assert_called_once_with(
            {'agent': 'macroeconomic_analysis_agent', 'insights': insights}
        )

class TestGeopoliticalRiskAgent(unittest.TestCase):
    def setUp(self):
        """Setup method to create an instance of GeopoliticalRiskAgent."""
        config = {'data_sources': {'news_api': Mock(), 'political_database': Mock()}}
        self.agent = GeopoliticalRiskAgent(config)

    @patch('core.agents.geopolitical_risk_agent.send_message')
    def test_assess_geopolitical_risks(self, mock_send_message):
        """Test assessing geopolitical risks."""
        #... (mock data sources and their responses)

        risk_assessments = self.agent.assess_geopolitical_risks()
        self.assertIsInstance(risk_assessments, dict)
        self.assertIn('political_risk_index', risk_assessments)
        self.assertIn('key_risks', risk_assessments)
        mock_send_message.assert_called_once_with(
            {'agent': 'geopolitical_risk_agent', 'risk_assessments': risk_assessments}
        )

#... (add test classes for other agents with similar structure)

if __name__ == '__main__':
    unittest.main()
