import unittest
import time
from unittest.mock import MagicMock, Mock

# Import the classes to be tested
from core.agents.orchestrators.workflow_manager import WorkflowManager
from core.agents.orchestrators.workflow import Workflow
from core.agents.orchestrators.task import Task
from core.agents.orchestrators.parallel_orchestrator import ParallelOrchestrator
from core.agents.orchestrators.credit_risk_orchestrator import CreditRiskOrchestrator

class TestWorkflowSystem(unittest.TestCase):

    def test_parallel_orchestrator(self):
        """
        Tests that the ParallelOrchestrator runs independent tasks in parallel.
        """
        orchestrator = ParallelOrchestrator(config={})
        num_tasks = 3
        longest_task_duration = num_tasks  # Durations are 1, 2, 3

        start_time = time.time()
        results = orchestrator.execute(num_tasks=num_tasks)
        execution_time = time.time() - start_time

        # Check that all tasks completed
        self.assertEqual(len(results['data']), num_tasks)

        # Check that execution was parallel (faster than sum of durations)
        # Sum of durations = 1 + 2 + 3 = 6
        self.assertLess(execution_time, sum(range(1, num_tasks + 2)))
        self.assertGreater(execution_time, longest_task_duration)

    def test_dependency_execution_order(self):
        """
        Tests that the WorkflowManager correctly handles task dependencies.
        """
        # Mock actions for dependent tasks
        task1_action = MagicMock(return_value="Data from Task 1")
        task2_action = MagicMock()

        # Define tasks, with task2 depending on task1
        task1 = Task(name="task1", action=task1_action)
        task2 = Task(name="task2", action=task2_action, dependencies=["task1"])

        workflow = Workflow(name="dependency_test", tasks=[task1, task2])
        manager = WorkflowManager(max_workers=2)
        results = manager.execute_workflow(workflow)

        # Check that the dependent task was called with the output of its dependency
        task2_action.assert_called_once_with(task1="Data from Task 1")

    def test_credit_risk_orchestrator_integration(self):
        """
        Tests the refactored CreditRiskOrchestrator with mocked sub-agents
        to ensure it correctly wires and executes a dependent workflow.
        """
        # Mock agents and their execute methods
        mock_news_agent = Mock()
        mock_news_agent.execute = MagicMock(return_value={"news_content": "bad news"})

        mock_sentiment_agent = Mock()
        # This mock expects the result of the news agent as a keyword argument
        mock_sentiment_agent.execute = MagicMock(return_value={"sentiment": "negative"})

        mock_config = {
            "sub_agents": {"financial_news_sub_agent": mock_news_agent},
            "meta_agents": {"sentiment_analysis_meta_agent": mock_sentiment_agent}
        }

        orchestrator = CreditRiskOrchestrator(config=mock_config)
        final_result = orchestrator.execute(query="Test Company")

        # Verify that the news agent was called with the initial query
        mock_news_agent.execute.assert_called_once_with(query="Test Company")

        # Verify that the sentiment agent was called with the result of the news agent
        mock_sentiment_agent.execute.assert_called_once_with(fetch_news={"news_content": "bad news"})

        # Verify the final synthesized result
        self.assertIn("fetch_news", final_result["data"])
        self.assertIn("analyze_sentiment", final_result["data"])
        self.assertEqual(final_result["data"]["analyze_sentiment"], {"sentiment": "negative"})

if __name__ == '__main__':
    unittest.main()