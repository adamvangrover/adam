import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure core can be imported
sys.path.append(os.getcwd())

from core.agents.anomaly_detection_agent import AnomalyDetectionAgent
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

class TestAnomalyDetectionAgent(unittest.TestCase):
    def setUp(self):
        # Combined config: Ticker is required for KG integration tests
        self.config = {"ticker": "AAPL"} 
        self.agent = AnomalyDetectionAgent(self.config)

    # --- Integration Tests (Knowledge Graph) ---

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
        self.assertGreater(first_revenue, 10000000) 

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

    # --- Logic Unit Tests (Calculations & Anomalies) ---

    def test_financial_ratios_calculation(self):
        # Create dummy data
        data = {
            'revenue': [1000.0, 1200.0],
            'net_income': [100.0, 150.0],
            'total_assets': [5000.0, 5500.0],
            'shareholders_equity': [2000.0, 2200.0],
            'current_assets': [1000.0, 1100.0],
            'current_liabilities': [500.0, 550.0],
            'inventory': [200.0, 220.0],
            'total_debt': [1000.0, 1100.0]
        }
        df = pd.DataFrame(data)

        ratios = self.agent._get_financial_ratios(df)

        self.assertIn('net_profit_margin', ratios.columns)
        self.assertIn('return_on_assets', ratios.columns)
        self.assertIn('return_on_equity', ratios.columns)
        self.assertIn('current_ratio', ratios.columns)
        self.assertIn('quick_ratio', ratios.columns)
        self.assertIn('debt_to_equity', ratios.columns)

        # Check values for first row
        self.assertAlmostEqual(ratios.iloc[0]['net_profit_margin'], 0.1)
        self.assertAlmostEqual(ratios.iloc[0]['current_ratio'], 2.0)

    def test_detect_company_anomalies(self):
        # Create data with an anomaly
        # Row 2 (index 2) has huge debt and low equity
        data = {
            'revenue': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0],
            'net_income': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            'total_assets': [5000.0, 5200.0, 5500.0, 5800.0, 6000.0, 6200.0, 6400.0, 6600.0],
            'shareholders_equity': [2000.0, 2100.0, 100.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0],
            'current_assets': [1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0],
            'current_liabilities': [500.0, 520.0, 550.0, 580.0, 600.0, 620.0, 640.0, 660.0],
            'inventory': [200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0],
            'total_debt': [1500.0, 1550.0, 6000.0, 1650.0, 1700.0, 1750.0, 1800.0, 1850.0]
        }
        # Overwrite internal data with test data
        self.agent.company_data = pd.DataFrame(data)

        anomalies = self.agent.detect_company_anomalies()

        found = any('debt_to_equity_anomaly' in a['type'] for a in anomalies)
        self.assertTrue(found, "Should detect debt_to_equity anomaly")

if __name__ == '__main__':
    unittest.main()