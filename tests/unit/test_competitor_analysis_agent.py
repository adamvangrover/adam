import unittest
from unittest.mock import MagicMock, patch
import asyncio
import sys
import os
import types

# Snapshot real modules so the mocks below don't leak into other test modules.
_MOCKED_MODULES = ['core.agents.agent_base', 'core.schemas.agent_schema']
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _MOCKED_MODULES}

# --- MOCKING DEPENDENCIES FOR ISOLATION ---
# Mock core.agents.agent_base
mock_agent_base_module = types.ModuleType("core.agents.agent_base")
class MockAgentBase:
    def __init__(self, config, **kwargs):
        self.config = config
    async def execute(self, *args, **kwargs):
        pass
mock_agent_base_module.AgentBase = MockAgentBase
sys.modules["core.agents.agent_base"] = mock_agent_base_module

# Mock core.schemas.agent_schema
mock_schema_module = types.ModuleType("core.schemas.agent_schema")
class AgentInput:
    def __init__(self, query, context=None):
        self.query = query
        self.context = context or {}
class AgentOutput:
    def __init__(self, answer, sources, confidence, metadata):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.metadata = metadata
mock_schema_module.AgentInput = AgentInput
mock_schema_module.AgentOutput = AgentOutput
sys.modules["core.schemas.agent_schema"] = mock_schema_module

# Now import the agent under test
# We use importlib to load it from file path, ensuring we bypass package init if needed,
# but since we mocked the modules it imports, standard import might work if path is set.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Direct import via file path to avoid triggering core.agents.__init__
import importlib.util
agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core/agents/competitor_analysis_agent.py'))
spec = importlib.util.spec_from_file_location("competitor_analysis_agent", agent_path)
caa_module = importlib.util.module_from_spec(spec)
sys.modules["competitor_analysis_agent"] = caa_module
spec.loader.exec_module(caa_module)

CompetitorAnalysisAgent = caa_module.CompetitorAnalysisAgent

for _name, _module in _ORIGINAL_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module



class TestCompetitorAnalysisAgent(unittest.TestCase):
    def setUp(self):
        self.agent = CompetitorAnalysisAgent(config={"agent_id": "test_competitor"})

    def test_execute_aapl(self):
        # Test standard input
        input_data = AgentInput(query="Analyze AAPL")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.agent.execute(input_data))
        loop.close()

        self.assertIsInstance(result, AgentOutput)
        self.assertIn("AAPL", result.answer)
        self.assertIn("MSFT", result.answer) # AAPL competitor
        self.assertIn("GOOGL", result.answer) # AAPL competitor
        self.assertEqual(result.metadata["target"], "AAPL")

    def test_execute_unknown(self):
        input_data = AgentInput(query="Analyze XYZ123")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.agent.execute(input_data))
        loop.close()

        self.assertIsInstance(result, AgentOutput)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("No competitors found", result.answer)

if __name__ == '__main__':
    unittest.main()
