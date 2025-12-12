
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from core.agents.anomaly_detection_agent import AnomalyDetectionAgent
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

class TestAnomalyDetectionAgent(unittest.TestCase):
    def setUp(self):
        self.config = {"ticker": "AAPL"}
        self.agent = AnomalyDetectionAgent(self.config)

    def test_load_company_data_integration(self):
        """
        Test that _load_company_data retrieves data derived from the KG
        (simulated time series based on seed data).
        """
        # This assumes AAPL exists in seed data and UnifiedKnowledgeGraph works
        df = self.agent._load_company_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('revenue', df.columns)
        self.assertIn('net_income', df.columns)
        self.assertEqual(len(df), 10)

        # Verify it's not the placeholder data (which starts at 1,000,000)
        # Based on seed AAPL (580B market cap), revenue should be much higher.
        first_revenue = df['revenue'].iloc[0]
        self.assertNotEqual(first_revenue, 1000000)
        self.assertGreater(first_revenue, 10000000) # Expecting ~30M+ based on my manual calculation

    @patch('core.agents.anomaly_detection_agent.UnifiedKnowledgeGraph')
    def test_load_company_data_fallback(self, mock_ukg_cls):
        """
        Test fallback to placeholder when KG fails or returns no data.
        """
        # Mock KG to return empty graph or raise error
        mock_kg = MagicMock()
        mock_kg.graph.nodes.return_value = [] # Empty iterator
        mock_ukg_cls.return_value = mock_kg

        agent = AnomalyDetectionAgent({"ticker": "UNKNOWN_CORP"})
        df = agent._load_company_data()

        # Should be placeholder
        self.assertEqual(df['revenue'].iloc[0], 1000000)

if __name__ == '__main__':
    unittest.main()
