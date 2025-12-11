
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
from core.agents.specialized.credit_conformance_agent import CreditConformanceAgent
from core.schemas.credit_conformance import CreditConformanceReport

class TestCreditConformanceAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.config = {
            "model_config": {
                "provider": "mock",
                "model": "gpt-4"
            }
        }
        self.agent = CreditConformanceAgent(self.config)
        self.agent.llm_plugin = MagicMock()
        # Mock generate_text instead of query
        # Since LLMPlugin.generate_text is synchronous in the base implementation I wrote (or wrapper),
        # but in agent.execute I called it synchronously: response_text = self.llm_plugin.generate_text(prompt)
        # However, the previous test setup used AsyncMock for query because legacy query might have been async.
        # Let's check CreditConformanceAgent.execute code again.
        # It calls `response_text = self.llm_plugin.generate_text(prompt)` (synchronous call).
        # So I should mock it as a regular Mock or MagicMock, not AsyncMock.
        self.agent.llm_plugin.generate_text = MagicMock()

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="Prompt with {{ document_under_review }} and {{ policy_standards }}")
    async def test_execute_success(self, mock_file):
        # Mock LLM Response
        mock_response = """
        ```json
        {
          "reportMetadata": {
            "documentReviewed": "Test Doc",
            "documentID": "123",
            "reviewDate": "2023-01-01",
            "reviewerPersona": "Credit Risk Control Officer",
            "overallConformanceStatus": "Full Conformance"
          },
          "findings": [
            {
              "status": "Conformant",
              "severityScore": "LOW",
              "confidenceScore": 0.99,
              "remediationAction": "None",
              "policyStandard": {
                "source": "Policy A",
                "clause": "1.1",
                "text": "Must be good."
              },
              "documentReference": {
                "source": "Doc A",
                "clause": "2.2",
                "text": "It is good."
              },
              "analysis": "Analysis here.",
              "verificationTrail": {
                "verificationQuestions": [
                    {"question": "Is it good?", "answer": "Yes."}
                ],
                "verificationOutcome": "Confirmed"
              }
            }
          ]
        }
        ```
        """
        self.agent.llm_plugin.generate_text.return_value = mock_response

        # Execute
        result = await self.agent.execute(document_text="Doc Text", policy_text="Policy Text")

        # Verify
        self.assertIsInstance(result, dict)
        self.assertEqual(result["reportMetadata"]["documentReviewed"], "Test Doc")
        self.assertEqual(result["findings"][0]["status"], "Conformant")

        # Verify Pydantic validation happened
        self.agent.llm_plugin.generate_text.assert_called_once()

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="Prompt")
    async def test_execute_validation_failure(self, mock_file):
        # Mock Invalid JSON
        self.agent.llm_plugin.generate_text.return_value = "Not JSON"

        # Execute
        result = await self.agent.execute(document_text="Doc", policy_text="Policy")

        # Verify error handling
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to generate valid report")

if __name__ == '__main__':
    unittest.main()
