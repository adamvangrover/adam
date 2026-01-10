# tests/verify_v23_remediation.py

import unittest
import logging
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner, PlannerIntent
from core.engine.agent_adapters import V23DataRetrieverAdapter, YFINANCE_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verifier")

class TestV23Remediation(unittest.TestCase):

    def setUp(self):
        self.planner = NeuroSymbolicPlanner()
        self.retriever = V23DataRetrieverAdapter()

    def test_planner_intent_classification(self):
        logger.info("Testing Semantic Classification...")

        req_dive = "I need a comprehensive deep dive into Apple's valuation."
        intent_dive = self.planner._classify_intent_semantic(req_dive)
        self.assertEqual(intent_dive, PlannerIntent.DEEP_DIVE, f"Failed DEEP_DIVE check. Got {intent_dive}")

        req_risk = "Alert me on any credit risk exposure for Tesla."
        intent_risk = self.planner._classify_intent_semantic(req_risk)
        self.assertEqual(intent_risk, PlannerIntent.RISK_ALERT, f"Failed RISK_ALERT check. Got {intent_risk}")

        req_market = "What is the market sentiment for Nvidia?"
        intent_market = self.planner._classify_intent_semantic(req_market)
        self.assertEqual(intent_market, PlannerIntent.MARKET_UPDATE, f"Failed MARKET_UPDATE check. Got {intent_market}")

        logger.info("Intent Classification Passed.")

    def test_planner_entity_extraction(self):
        logger.info("Testing Dynamic Entity Extraction...")

        req_1 = "Analyze NVDA credit rating"
        ent_1 = self.planner._extract_entities_dynamic(req_1)
        self.assertEqual(ent_1.get("primary_entity"), "NVDA", f"Failed extraction for NVDA. Got {ent_1}")

        req_2 = "What is the risk for TSLA?" # Heuristic check
        ent_2 = self.planner._extract_entities_dynamic(req_2)
        # Assuming heuristic picks uppercased TSLA
        self.assertEqual(ent_2.get("primary_entity"), "TSLA", f"Failed extraction for TSLA. Got {ent_2}")

        logger.info("Entity Extraction Passed.")

    def test_data_retriever_live(self):
        logger.info("Testing Live Data Retrieval...")

        if not YFINANCE_AVAILABLE:
            logger.warning("YFinance not available. Skipping live test.")
            return

        # Fetch Microsoft data
        data = self.retriever.get_financials("MSFT")

        self.assertIsNotNone(data, "Data should not be None")
        self.assertIn("company_info", data, "Missing company_info")

        # Check if it looks like real data (name should contain Microsoft)
        name = data["company_info"].get("name", "")
        self.assertIn("Microsoft", name, f"Expected Microsoft in name, got {name}")

        # Check financial fields
        fin = data.get("financial_data_detailed", {})
        self.assertTrue(len(fin.get("income_statement", {}).get("revenue", [])) > 0, "Revenue data missing")

        logger.info("Live Data Retrieval Passed.")

    def test_data_retriever_fallback(self):
        logger.info("Testing Data Retrieval Fallback...")

        # Use a fake ticker that yfinance should fail on
        data = self.retriever.get_financials("INVALID_TICKER_XYZ_123")

        # Should fallback to mock data which constructs a name
        self.assertIsNotNone(data, "Should return mock data on failure")
        self.assertIn("INVALID_TICKER_XYZ_123 Corp", data["company_info"]["name"], "Mock name mismatch")

        logger.info("Fallback Passed.")

if __name__ == "__main__":
    unittest.main()
