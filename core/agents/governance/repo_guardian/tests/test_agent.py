import unittest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.governance.repo_guardian.agent import RepoGuardianAgent
from core.agents.governance.repo_guardian.schemas import PullRequest, FileDiff, ReviewDecisionStatus

class TestRepoGuardianAgent(unittest.TestCase):
    def setUp(self):
        self.config = {"some": "config"}
        self.agent = RepoGuardianAgent(config=self.config)

    def test_initialization(self):
        self.assertIsNotNone(self.agent.tools)
        self.assertIsNotNone(self.agent.analyzer)
        self.assertIsNotNone(self.agent.scanner)

    def test_run_heuristics_security(self):
        pr = PullRequest(
            pr_id="PR-1",
            author="test_user",
            title="Bad PR",
            description="Adds a secret",
            files=[
                FileDiff(
                    filepath="config.py",
                    change_type="add",
                    diff_content="api_key = 'AKIA1234567890123456'"
                )
            ]
        )
        comments, results = self.agent._run_heuristics(pr)
        self.assertTrue(any(c.severity == "critical" and "SECURITY" in c.message for c in comments))
        self.assertTrue(len(results["config.py"].security_findings) > 0)

    def test_run_heuristics_python_ast(self):
        pr = PullRequest(
            pr_id="PR-2",
            author="test_user",
            title="Code PR",
            description="Adds a function",
            files=[
                FileDiff(
                    filepath="main.py",
                    change_type="add",
                    diff_content="def new_func(): pass", # Not used for AST if new_content exists, but used for fallback
                    new_content="def new_func():\n    pass"
                )
            ]
        )
        comments, results = self.agent._run_heuristics(pr)
        # Should have missing docstring and type hints in results
        self.assertTrue(len(results["main.py"].missing_docstrings) > 0)
        self.assertTrue(len(results["main.py"].missing_type_hints) > 0)

    @patch('core.agents.governance.repo_guardian.agent.RepoGuardianAgent._llm_review')
    def test_execute_critical_security_failure(self, mock_llm):
        # Setup mock to return an approval (to show it gets overridden)
        async def mock_review(*args, **kwargs):
            from core.agents.governance.repo_guardian.schemas import ReviewDecision
            return ReviewDecision(
                pr_id="PR-1",
                status="approve",
                summary="LGTM",
                score=100
            )
        mock_llm.side_effect = mock_review

        pr_data = {
            "pr_id": "PR-1",
            "author": "hacker",
            "title": "Hack",
            "description": "...",
            "files": [
                {
                    "filepath": "secrets.py",
                    "change_type": "add",
                    "diff_content": "key = 'AKIA1234567890123456'"
                }
            ]
        }

        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        decision = loop.run_until_complete(self.agent.execute(pr=pr_data))
        loop.close()

        # Should be rejected due to critical security heuristic despite LLM approval
        self.assertEqual(decision.status, ReviewDecisionStatus.REJECT)
        self.assertTrue("Decision downgraded" in decision.summary)
        self.assertTrue(decision.score < 80)

if __name__ == '__main__':
    unittest.main()
