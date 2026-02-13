import unittest
import json
import sys
import os

# Add the directory containing the module to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../enterprise_bundle/adam-sovereign-bundle/governance')))

from adr_controls import audit_citation_density, audit_financial_math

class TestAdrControls(unittest.TestCase):
    def test_audit_citation_density_pass(self):
        # 1 citation per 2 sentences. 2 sentences, 1 citation = 0.5 density. Pass.
        text = "This is a sentence. This is another sentence with a citation [doc_1:chunk_1]."
        passed, message = audit_citation_density(text)
        self.assertTrue(passed)
        self.assertEqual(message, "Passed")

    def test_audit_citation_density_fail(self):
        # 3 sentences, 1 citation = 0.33 density. Fail.
        text = "This is a sentence. This is another sentence. This one has a citation [doc_1:chunk_1]."
        passed, message = audit_citation_density(text)
        self.assertFalse(passed)
        self.assertEqual(message, "Insufficient Evidence Linking")

    def test_audit_financial_math_pass(self):
        data = {
            "total_assets": 100,
            "total_liabilities": 50,
            "total_equity": 50
        }
        passed, message = audit_financial_math(json.dumps(data))
        self.assertTrue(passed)
        self.assertEqual(message, "Passed")

    def test_audit_financial_math_fail(self):
        data = {
            "total_assets": 100,
            "total_liabilities": 40,
            "total_equity": 50
        }
        passed, message = audit_financial_math(json.dumps(data))
        self.assertFalse(passed)
        self.assertIn("Balance Sheet Mismatch", message)

if __name__ == '__main__':
    unittest.main()
