import unittest
import sys
import os
import pandas as pd

# Add core to path
sys.path.append(os.getcwd())

from core.agents.anomaly_detection_agent import AnomalyDetectionAgent

class TestAnomalyDetectionAgent(unittest.TestCase):
    def setUp(self):
        self.config = {}
        self.agent = AnomalyDetectionAgent(self.config)

    def test_detect_company_anomalies(self):
        anomalies = self.agent.detect_company_anomalies()

        self.assertIsInstance(anomalies, list)
        self.assertTrue(len(anomalies) > 0, "No anomalies detected")

        methods = set(a['method'] for a in anomalies)
        # We expect these methods to be used
        expected_methods = {'z-score', 'LOF', 'One-Class SVM', 'z-score on ratio'}

        # Check if at least some of the expected methods are present
        found_intersection = methods.intersection(expected_methods)
        self.assertTrue(len(found_intersection) > 0, f"Expected methods {expected_methods} not found in {methods}")

        # Check types of anomalies
        types = set(a['type'] for a in anomalies)
        self.assertIn('company_data_anomaly', types)
        self.assertIn('financial_ratio_anomaly', types)

if __name__ == '__main__':
    unittest.main()
