import json
import unittest

from core.prompting.personas.risk_officer import RiskOfficerPersona


class TestRiskOfficerPersona(unittest.TestCase):
    def test_rendering_without_optional_fields(self):
        persona = RiskOfficerPersona.default()
        inputs = {"draft_analysis": "Revenue grew but we have High Insolvency Risk.", "ticker": "AAPL", "iteration": 1}
        messages = persona.render_messages(inputs)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Senior Risk Officer (SRO)", messages[0]["content"])
        self.assertIn("Hallucination Check", messages[0]["content"])

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("AAPL", messages[1]["content"])
        self.assertIn("Revenue grew", messages[1]["content"])
        self.assertNotIn("FINANCIAL CONTEXT", messages[1]["content"])
        self.assertNotIn("POLICY CONSTRAINTS", messages[1]["content"])

    def test_rendering_with_optional_fields(self):
        persona = RiskOfficerPersona.default()
        inputs = {
            "draft_analysis": "Revenue is up.",
            "ticker": "AAPL",
            "iteration": 2,
            "financial_context": "EBITDA down 20%",
            "policy_constraints": "DSCR > 1.25x"
        }
        messages = persona.render_messages(inputs)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("FINANCIAL CONTEXT", messages[1]["content"])
        self.assertIn("EBITDA down 20%", messages[1]["content"])
        self.assertIn("POLICY CONSTRAINTS", messages[1]["content"])
        self.assertIn("DSCR > 1.25x", messages[1]["content"])

    def test_parsing(self):
        persona = RiskOfficerPersona.default()
        mock_response = json.dumps({
            "status": "FAIL",
            "quality_score": 0.5,
            "missing_elements": ["Liquidity Analysis"],
            "logical_flaws": ["Bullish but High Insolvency Risk"],
            "policy_breaches": ["DSCR dropped below 1.25x"],
            "instructions": "Fix the flaws."
        })

        parsed = persona.parse_response(mock_response)
        self.assertEqual(parsed.status, "FAIL")
        self.assertEqual(parsed.quality_score, 0.5)
        self.assertEqual(len(parsed.policy_breaches), 1)
        self.assertEqual(parsed.policy_breaches[0], "DSCR dropped below 1.25x")

if __name__ == "__main__":
    unittest.main()
