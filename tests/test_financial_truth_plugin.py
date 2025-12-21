from __future__ import annotations

import unittest

from core.prompting.base_prompt_plugin import PromptMetadata
from core.prompting.loader import PromptLoader
from core.prompting.plugins.financial_truth_plugin import FinancialTruthPlugin


class TestFinancialTruthPlugin(unittest.TestCase):
    def setUp(self):
        # Load the prompt template content
        # We need to construct the plugin manually since it's not loading from a YAML config yet
        # and BasePromptPlugin expects template strings or engines.

        # In a real app, we might load this from the markdown file we just created.
        prompt_content = PromptLoader.get("AOPL-v1.0/professional_outcomes/LIB-PRO-009_financial_truth_tao")

        # The prompt content is the full markdown.
        # For the plugin, we treat it as a "user_template" effectively,
        # or split it if we wanted strict system/user separation.
        # Given the file structure, it's one big block. We'll pass it as user_template.

        self.metadata = PromptMetadata(
            prompt_id="LIB-PRO-009",
            version="1.0",
            author="Adam v23.5",
            model_config={"temperature": 0.0} # Low temp for auditing
        )

        self.plugin = FinancialTruthPlugin(
            metadata=self.metadata,
            user_template=prompt_content
        )

    def test_rendering(self):
        inputs = {
            "context": "Apple Inc. reported revenue of $80 billion in Q1 2024.",
            "question": "What was Apple's revenue?"
        }
        messages = self.plugin.render_messages(inputs)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Apple Inc. reported revenue of $80 billion", messages[0]["content"])
        self.assertIn("What was Apple's revenue?", messages[0]["content"])
        # Matched casing to actual prompt text
        self.assertIn("Your goal is to answer the User's Question based **SOLELY** on the provided", messages[0]["content"])

    def test_response_parsing(self):
        # Simulate a valid LLM response
        raw_response = """
<thinking>
1. Scanning for revenue. Found.
</thinking>
**Answer:** Apple's revenue was $80 billion.
**Evidence:** "Apple Inc. reported revenue of $80 billion in Q1 2024."
**Logic:** Extracted revenue figure from the Q1 report sentence.
        """

        output = self.plugin.parse_response(raw_response)

        self.assertEqual(output.answer, "Apple's revenue was $80 billion.")
        self.assertEqual(output.evidence, '"Apple Inc. reported revenue of $80 billion in Q1 2024."')
        self.assertEqual(output.logic, "Extracted revenue figure from the Q1 report sentence.")

    def test_response_parsing_failure(self):
        # Simulate an invalid response
        raw_response = "I don't know."

        # The regex parser should return "Parse Error" strings rather than crashing,
        # per my implementation choice (or it could raise, but let's check what I wrote).
        output = self.plugin.parse_response(raw_response)

        self.assertIn("Parse Error", output.answer)

if __name__ == "__main__":
    unittest.main()
