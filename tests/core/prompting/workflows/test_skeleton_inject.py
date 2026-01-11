import sys
import os
import unittest
import json
from unittest.mock import MagicMock

# Ensure repo root is in python path
sys.path.append(os.getcwd())

from core.prompting.workflows.skeleton_inject import SkeletonInjectWorkflow, JSONFileFetcher

class MockLLMClient:
    def generate(self, messages):
        content = ""
        user_content = ""
        for m in messages:
            if m["role"] == "user":
                user_content += m["content"]

        # Debugging
        # print(f"DEBUG: User Content: {user_content}")

        if "We are drafting the \"Financial Performance & Outlook\"" in user_content:
            return """
Top-line performance was {{REVENUE_DIRECTION}} year-over-year, settling at {{REVENUE_CURRENT}}. This variance of {{REVENUE_YOY_VAR}} was primarily driven by strategic pricing actions. EBITDA margins {{MARGIN_DIRECTION}} to {{EBITDA_MARGIN}}.
            """
        elif "You are the \"Editor\" validating a draft Credit Memo" in user_content:
            return """
Top-line performance was grew modestly year-over-year, settling at $12.4B. This variance of +5% was primarily driven by strategic pricing actions. EBITDA margins expanded to 25.8%.
            """
        elif "A junior analyst has submitted the following" in user_content: # Phase 3
            return json.dumps({
                "status": "APPROVED",
                "score": 95,
                "feedback": "Solid analysis, consistent with data.",
                "red_flags": []
            })

        return "Unknown Prompt"

class TestSkeletonInjectWorkflow(unittest.TestCase):
    def test_workflow_execution_with_critique(self):
        print("Initializing Skeleton & Inject Workflow Test (Full Flow)...")

        mock_llm = MockLLMClient()
        data_path = os.path.join(os.getcwd(), "data/analyst_os_demo_data.json")
        fetcher = JSONFileFetcher(data_path)

        workflow = SkeletonInjectWorkflow(llm_client=mock_llm, data_fetcher=fetcher, tone="Hawkish")

        context_chunks = "Mock earnings transcript."

        result = workflow.run(context_chunks)

        print(f"Final Report: {result.final_text}")
        print(f"Critique: {result.critique}")

        self.assertIn("$12.4B", result.final_text)
        self.assertIsNotNone(result.critique)
        self.assertEqual(result.critique["status"], "APPROVED")
        self.assertIn("phase_1_output", result.audit_trace)

if __name__ == "__main__":
    unittest.main()
