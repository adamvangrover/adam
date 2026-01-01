import unittest
from core.engine.risk_consensus_engine import RiskConsensusEngine

class TestRiskConsensus(unittest.TestCase):

    def setUp(self):
        self.engine = RiskConsensusEngine(alpha=0.4, beta=0.4, gamma=0.5)

    def test_consensus_pass(self):
        # Both Agree
        metric = self.engine.calculate_consensus("Pass", "Pass", 0.95)

        # Score = 0.4(1) + 0.4(0.95) - 0 = 0.4 + 0.38 = 0.78
        self.assertAlmostEqual(metric.conviction_score, 0.78)
        self.assertEqual(metric.final_rating, "Pass")
        self.assertIn("Consensus Reached", metric.narrative)

    def test_regulatory_constraint(self):
        # Reg says Fail (Substandard), Strat says Pass
        # Diff = |3 - 1| = 2. Penalty = (2/4)*0.5 = 0.25
        metric = self.engine.calculate_consensus("Substandard", "Pass", 0.90)

        # Score = 0 + 0.4(0.90) - 0.25 = 0.36 - 0.25 = 0.11 (Low Conviction)
        self.assertAlmostEqual(metric.conviction_score, 0.11)
        # Should adopt the stricter rating
        self.assertEqual(metric.final_rating, "Substandard")
        self.assertIn("Adopted Regulatory Rating", metric.narrative)

    def test_hidden_risk(self):
        # Reg says Pass, Strat says Fail (Doubtful)
        # Diff = |1 - 4| = 3. Penalty = (3/4)*0.5 = 0.375
        metric = self.engine.calculate_consensus("Pass", "Doubtful", 0.90)

        # Score = 0 + 0.36 - 0.375 = -0.015 -> 0.0
        self.assertEqual(metric.conviction_score, 0.0)
        # Should adopt the stricter rating (Strategic in this case)
        self.assertEqual(metric.final_rating, "Doubtful")
        self.assertIn("Adopted Strategic Rating", metric.narrative)

if __name__ == '__main__':
    unittest.main()
