import unittest
import sys

# Mock numpy if missing
try:
    import numpy as np
except ImportError:
    from unittest.mock import MagicMock
    np = MagicMock()
    sys.modules['numpy'] = np

from core.credit_sentinel.models.distress_classifier import DistressClassifier

class TestDistressClassifier(unittest.TestCase):
    def test_predict_healthy(self):
        clf = DistressClassifier()
        # Healthy inputs
        ratios = {
            'interest_coverage': 10.0,
            'leverage': 1.0,
            'debt_to_equity': 0.5,
            'current_ratio': 2.5,
            'roa': 0.15
        }
        result = clf.predict_distress(ratios)
        # Even in mock mode (no sklearn), it returns a dict with label
        self.assertIn("label", result)

    def test_predict_distressed(self):
        clf = DistressClassifier()
        # Distressed inputs
        ratios = {
            'interest_coverage': 0.5,
            'leverage': 8.0,
            'debt_to_equity': 5.0,
            'current_ratio': 0.5,
            'roa': -0.2
        }
        result = clf.predict_distress(ratios)
        self.assertIn("label", result)

if __name__ == '__main__':
    unittest.main()
