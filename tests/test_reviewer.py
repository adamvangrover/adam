import unittest
from core.v30_architecture.python_intelligence.agents.reviewer_agent import ReviewerAgent

class TestReviewerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReviewerAgent()

    def test_consistent_report(self):
        content = {"summary": "The market is rallying with strong bullish momentum."}
        tech_data = {"signal": "BULLISH", "conviction": 85, "price": 100.0}

        result = self.agent.review("Market_Pulse", content, tech_data)

        print(f"\nConsistent Review: {result}")
        self.assertTrue(result['quality_score'] > 90)
        self.assertNotIn("sentiment_mismatch", result['flags'])

    def test_inconsistent_report(self):
        content = {"summary": "We see a massive crash coming. Bearish outlook."}
        tech_data = {"signal": "BULLISH", "conviction": 85, "price": 100.0}

        result = self.agent.review("Market_Pulse", content, tech_data)

        print(f"\nInconsistent Review: {result}")
        self.assertTrue(result['quality_score'] < 90)
        self.assertIn("sentiment_mismatch", result['flags'])

if __name__ == "__main__":
    unittest.main()
