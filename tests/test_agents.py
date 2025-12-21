# tests/test_agents.py

import unittest
from unittest.mock import Mock, patch

from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.market_sentiment_agent import MarketSentimentAgent

#... (other imports)

class TestMarketSentimentAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Setup method to create an instance of MarketSentimentAgent."""
        config = {
            'data_sources': [],
            'sentiment_threshold': 0.6
        }
        self.agent = MarketSentimentAgent(config)
        # Mock APIs
        self.agent.news_api = Mock()
        self.agent.prediction_market_api = Mock()
        self.agent.social_media_api = Mock()
        self.agent.web_traffic_api = Mock()

    async def test_analyze_sentiment(self):
        """Test analyzing market sentiment."""
        self.agent.news_api.get_headlines.return_value = (["Headline"], 0.5)
        self.agent.prediction_market_api.get_market_sentiment.return_value = 0.5
        self.agent.social_media_api.get_social_media_sentiment.return_value = 0.5
        self.agent.web_traffic_api.get_web_traffic_sentiment.return_value = 0.5

        sentiment, details = await self.agent.analyze_sentiment()
        self.assertAlmostEqual(sentiment, 0.5)

    async def test_analyze_sentiment_with_positive_news(self):
        """Test analyzing market sentiment with positive news."""
        self.agent.news_api.get_headlines.return_value = (["Positive"], 1.0)
        self.agent.prediction_market_api.get_market_sentiment.return_value = 0.0
        self.agent.social_media_api.get_social_media_sentiment.return_value = 0.0
        self.agent.web_traffic_api.get_web_traffic_sentiment.return_value = 0.0

        sentiment, details = await self.agent.analyze_sentiment()
        # 1.0 * 0.4 = 0.4
        self.assertAlmostEqual(sentiment, 0.4)

    async def test_analyze_sentiment_with_negative_news(self):
        """Test analyzing market sentiment with negative news."""
        self.agent.news_api.get_headlines.return_value = (["Negative"], -1.0)
        self.agent.prediction_market_api.get_market_sentiment.return_value = 0.0
        self.agent.social_media_api.get_social_media_sentiment.return_value = 0.0
        self.agent.web_traffic_api.get_web_traffic_sentiment.return_value = 0.0

        sentiment, details = await self.agent.analyze_sentiment()
        # -1.0 * 0.4 = -0.4
        self.assertAlmostEqual(sentiment, -0.4)

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
        self.agent.data_sources['government_stats_api'].get_gdp.return_value = None # Completed assignment
        self.agent.data_sources['government_stats_api'].get_inflation.return_value = None # Completed assignment

        insights = self.agent.analyze_macroeconomic_data()
        self.assertIsInstance(insights, dict)  # Check if insights is a dictionary
        self.assertIn('GDP_growth_trend', insights)
        self.assertIn('inflation_outlook', insights)
        mock_send_message.assert_called_once_with(
            {'agent': 'macroeconomic_analysis_agent', 'insights': insights}
        )

    @patch('core.agents.macroeconomic_analysis_agent.send_message')
    def test_analyze_macroeconomic_data_with_high_gdp_growth(self, mock_send_message):
        """Test analyzing macroeconomic data with high GDP growth."""
        # Mock the data source's responses with high GDP growth
        self.agent.data_sources['government_stats_api'].get_gdp.return_value = {'value': 2.5}
        self.agent.data_sources['government_stats_api'].get_inflation.return_value = {'value': 3.1}

        insights = self.agent.analyze_macroeconomic_data()
        self.assertEqual(insights['GDP_growth_trend'], 'positive')  # Check if GDP growth trend is positive


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
