import unittest
import asyncio
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure core path is in sys.path
sys.path.append(os.getcwd())

# Mock heavy dependencies to ensure test speed and stability
for mod in ["neo4j", "tensorflow", "torch", "transformers", "spacy", "pandas", "numpy", "networkx",
            "dotenv", "semantic_kernel", "semantic_kernel.connectors", "semantic_kernel.connectors.ai",
            "semantic_kernel.connectors.ai.open_ai", "pika", "redis", "boto3", "google.cloud", "tiktoken",
            "textblob", "shap", "lime", "dowhy", "matplotlib", "seaborn", "matplotlib.pyplot",
            "tweepy", "scipy", "scipy.stats", "sklearn", "sklearn.ensemble", "sklearn.model_selection", "sklearn.metrics",
             "ta", "talib", "bs4", "reportlab", "reportlab.lib.pagesizes", "reportlab.pdfgen", "fpdf",
             "aiohttp", "feedparser", "mesa", "psycopg2", "rdflib", "prometheus_client", "flask", "flask_socketio",
             "langchain.utilities", "langchain_community", "langchain_community.utilities"]:
    sys.modules[mod] = MagicMock()

from core.engine.meta_orchestrator import MetaOrchestrator
from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
# from core.engine.states import GraphState, PlanOnGraph # Imports might trigger dependencies, relying on mocks inside classes

# Helper for async mocks
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

class TestV23Architect(unittest.TestCase):
    def setUp(self):
        # Mock legacy orchestrator to avoid loading full agent system
        self.mock_legacy = MagicMock()
        # Mock the internal planner creation to avoid UnifiedKnowledgeGraph loading
        with patch("core.engine.meta_orchestrator.NeuroSymbolicPlanner") as MockPlanner:
            self.meta = MetaOrchestrator(legacy_orchestrator=self.mock_legacy)
            self.mock_planner_instance = MockPlanner.return_value
            self.meta.planner = self.mock_planner_instance

    def test_planner_logic(self):
        """Test the updated NeuroSymbolicPlanner logic directly."""
        # We instantiate a real planner here, but mock its KG
        with patch("core.engine.neuro_symbolic_planner.UnifiedKnowledgeGraph") as MockKG:
            planner = NeuroSymbolicPlanner()
            planner.kg = MockKG.return_value
            planner.kg.graph = MagicMock()

            # Mock shortest_path to return a valid path
            # We patch networkx where it is imported in the module
            with patch("core.engine.neuro_symbolic_planner.nx.shortest_path", return_value=["StartNode", "MidNode", "EndNode"]):
                # Also mock get_edge_data on the graph object
                planner.kg.graph.get_edge_data.return_value = {"relation": "connected_to"}

                plan = planner.discover_plan("StartNode", "EndNode")

            self.assertIsNotNone(plan)
            self.assertEqual(len(plan["steps"]), 2) # 2 edges
            self.assertIn("MATCH path =", plan["symbolic_plan"])

    def test_meta_orchestrator_routing_high(self):
        """Test routing to Adaptive Flow (HIGH complexity)."""
        async def run_test():
            # Setup Mocks
            self.mock_planner_instance.discover_plan.return_value = {
                "steps": [{"description": "test step"}],
                "symbolic_plan": "MATCH (n) RETURN n"
            }

            mock_app = MagicMock()
            mock_app.ainvoke = AsyncMock(return_value={"human_readable_status": "Success"})
            self.mock_planner_instance.to_executable_graph.return_value = mock_app

            # Patch _reflect_on_result to verify it's called
            with patch.object(self.meta, '_reflect_on_result', new_callable=AsyncMock) as mock_reflect:
                mock_reflect.return_value = {"status": "Reflected"}

                # Execute
                result = await self.meta.route_request("Analyze complex risk for Apple", context={})

                # Verify
                # The query "Analyze complex risk..." should trigger HIGH complexity logic
                # (via "analyze", "risk" keywords) -> _run_adaptive_flow

                self.mock_planner_instance.discover_plan.assert_called()
                self.mock_planner_instance.to_executable_graph.assert_called()
                mock_reflect.assert_called()

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
