from core.agents.agent_base import AgentBase
from .workflow_manager import WorkflowManager
from .workflow import Workflow
from .task import Task
import asyncio


class CreditRiskOrchestrator(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.sub_agents = self.config.get("sub_agents", {})
        self.meta_agents = self.config.get("meta_agents", {})
        self.workflow_manager = WorkflowManager()

    async def execute(self, query):
        # 1. Decompose the query into a workflow
        workflow = self._create_workflow(query)

        # 2. Execute the workflow
        # Assuming workflow_manager is updated to handle async tasks, or we wrap it here
        # For now, let's assume workflow_manager is synchronous but invokes actions that might be async
        # We need to ensure we await results if they are coroutines.

        # NOTE: Since WorkflowManager likely executes tasks sequentially or in threads,
        # mixing async actions directly might be tricky without updating WorkflowManager.
        # But for this specific test case, let's try to handle the async actions manually
        # or assume WorkflowManager can handle them if they were updated.

        # However, checking `test_workflow_system.py`, it seems WorkflowManager isn't async aware by default.
        # Let's inspect WorkflowManager next. But to fix the immediate test failure:

        # If we look at the failure: `mock_sentiment_agent.execute` calls receive a coroutine object
        # as argument for `fetch_news`. This means the result of `financial_news_agent.execute` was NOT awaited
        # before being passed to the next task.

        results = self.workflow_manager.execute_workflow(workflow)

        # Since we are moving to Async Native, we should ideally rewrite WorkflowManager to be async.
        # But for "minimal structural edits", let's handle the awaiting here if possible,
        # OR update WorkflowManager to await coroutines.

        # Given the "Async Native" core philosophy, updating WorkflowManager is the correct path.

        # 3. Synthesize the results
        return self._synthesize_results(results)

    def _create_workflow(self, query):
        financial_news_agent = self.sub_agents.get("financial_news_sub_agent")
        sentiment_analysis_agent = self.meta_agents.get("sentiment_analysis_meta_agent")

        tasks = []
        if financial_news_agent and sentiment_analysis_agent:
            task1 = Task(
                name="fetch_news",
                action=financial_news_agent.execute,
                query=query
            )

            # The sentiment analysis task depends on the news fetching task.
            # The result of 'fetch_news' will be passed as a keyword argument 'fetch_news'
            # to the `execute` method of the sentiment analysis agent.
            task2 = Task(
                name="analyze_sentiment",
                action=sentiment_analysis_agent.execute,
                dependencies=["fetch_news"]
            )

            tasks = [task1, task2]

        return Workflow(name="credit_risk_workflow", tasks=tasks)

    def _synthesize_results(self, results):
        # In a real implementation, this would use an LLM to synthesize the results
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.9,
            "data": results,
        }
