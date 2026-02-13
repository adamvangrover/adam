import unittest
import sys
import os
import json

# Add parent directory to path to import governance module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from governance.adr_controls import audit_citation_density, audit_financial_math, audit_tone_check, audit_absolute_statements

class TestGovernanceControls(unittest.TestCase):

    def test_audit_citation_density(self):
        # Case 1: Good density (100% citation)
        text_good = "This is a sentence. [doc_1] This is another sentence. [doc_2]"
        self.assertTrue(audit_citation_density(text_good)[0])

        # Case 2: Bad density (0% citation)
        text_bad = "This is a sentence. This is another sentence. No citations here."
        self.assertFalse(audit_citation_density(text_bad)[0])

    def test_audit_financial_math(self):
        # Case 1: Balanced
        json_good = json.dumps({"total_assets": 100, "total_liabilities": 50, "total_equity": 50})
        self.assertTrue(audit_financial_math(json_good)[0])

        # Case 2: Unbalanced
        json_bad = json.dumps({"total_assets": 100, "total_liabilities": 50, "total_equity": 40})
        self.assertFalse(audit_financial_math(json_bad)[0])

    def test_audit_tone_check(self):
        # Case 1: Professional
        text_good = "The revenue increased by 10%."
        self.assertTrue(audit_tone_check(text_good)[0])

        # Case 2: Unprofessional
        text_bad = "The revenue skyrocketed like crazy!"
        self.assertFalse(audit_tone_check(text_bad)[0])
        self.assertIn("skyrocketed", audit_tone_check(text_bad)[1])

    def test_audit_absolute_statements(self):
        # Case 1: Measured
        text_good = "It is likely that the trend will continue."
        self.assertTrue(audit_absolute_statements(text_good)[0])

        # Case 2: Absolute
        text_bad = "It is guaranteed to succeed."
        self.assertFalse(audit_absolute_statements(text_bad)[0])
        self.assertIn("guaranteed", audit_absolute_statements(text_bad)[1])

        # Case 3: False positive check (substring)
        text_fp = "Whenever we check, it is fine."
        # Should not flag "never" inside "whenever"
        self.assertTrue(audit_absolute_statements(text_fp)[0])

if __name__ == '__main__':
    unittest.main()
