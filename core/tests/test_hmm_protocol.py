import unittest
from core.system.hmm_protocol import HMMParser

class TestHMMProtocol(unittest.TestCase):

    def test_parse_request(self):
        text = """
HMM INTERVENTION REQUEST
 * Request ID: 2025-10-28-CR-001
 * Action: OVERRIDE_COVENANT_DEFINITION
 * Target: Net Leverage Ratio
 * Justification: The credit agreement amendment dated 2025-09-15 excludes "One-Time Restructuring Costs".
 * Parameters:
   * add_back_cap: 50,000,000
   * item_type: restructuring_costs
        """
        data = HMMParser.parse_request(text)
        self.assertEqual(data['action'], "OVERRIDE_COVENANT_DEFINITION")
        self.assertEqual(data['target'], "Net Leverage Ratio")
        self.assertEqual(data['parameters']['add_back_cap'], 50000000.0)
        self.assertEqual(data['parameters']['item_type'], "restructuring_costs")

    def test_generate_log(self):
        log = HMMParser.generate_log(
            action_taken="Updated Logic",
            impact_analysis={"Old Headroom": "5%", "New Headroom": "12%"},
            audit_link="doc_123.pdf",
            log_id="LOG-001"
        )
        self.assertIn("HMM ACTION LOG", log)
        self.assertIn(" * Log ID: LOG-001", log)
        self.assertIn(" * Action Taken: Updated Logic", log)
        self.assertIn("   * Old Headroom: 5%", log)

    def test_generate_request(self):
        req = HMMParser.generate_request(
            action="TEST_ACTION",
            target="TEST_TARGET",
            justification="Because test",
            parameters={"p1": 100},
            request_id="REQ-001"
        )
        self.assertIn("HMM INTERVENTION REQUEST", req)
        self.assertIn(" * Request ID: REQ-001", req)
        self.assertIn("   * p1: 100", req)

if __name__ == '__main__':
    unittest.main()
