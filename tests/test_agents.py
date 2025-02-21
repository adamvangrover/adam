# tests/test_agents.py

import unittest
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalyst
from core.agents.technical_analyst_agent import TechnicalAnalyst
from core.agents.risk_assessment_agent import RiskAssessor
from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialist
from core.agents.data_verification_agent import DataVerificationAgent
from core.agents.lexica_agent import LexicaAgent
from core.agents.archive_manager_agent import ArchiveManagerAgent

class TestMarketSentimentAgent(unittest.TestCase):
    #... (existing code)

    def test_analyze_sentiment(self):
        """Test analyzing market sentiment."""
        sentiment = self.agent.analyze_sentiment()
        self.assertIsNotNone(sentiment)
        self.assertIn(sentiment, ["positive", "negative", "neutral"])  # Check if sentiment is valid

class TestMacroeconomicAnalysisAgent(unittest.TestCase):
    #... (existing code)

    def test_analyze_macroeconomic_data(self):
        """Test analyzing macroeconomic data."""
        data = self.agent.analyze_macroeconomic_data()
        self.assertIsNotNone(data)
        self.assertIn('GDP_growth', data)  # Check if GDP growth is present
        self.assertIn('inflation', data)  # Check if inflation is present

class TestGeopoliticalRiskAgent(unittest.TestCase):
    #... (existing code)

    def test_assess_geopolitical_risks(self):
        """Test assessing geopolitical risks."""
        risks = self.agent.assess_geopolitical_risks()
        self.assertIsNotNone(risks)
        self.assertIsInstance(risks, list)  # Check if risks is a list

#... (add test classes for other agents with similar structure)

if __name__ == '__main__':
    unittest.main()
