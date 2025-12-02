import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.engine.meta_orchestrator import MetaOrchestrator

@pytest.mark.asyncio
async def test_deep_dive_pipeline():
    # Mock LLMPlugin to avoid API key requirements and AgentOrchestrator initialization issues
    with patch('core.system.agent_orchestrator.LLMPlugin') as MockLLM:
        orchestrator = MetaOrchestrator()

        # We trigger the "Deep Dive" path
        query = "Perform a deep dive analysis on TestCorp"
        context = {"simulation_depth": "Deep"}

        result = await orchestrator.route_request(query, context)

        assert result is not None
        if "error" in result:
            pytest.fail(f"Pipeline returned error: {result['error']}")

        assert "v23_knowledge_graph" in result

        kg = result["v23_knowledge_graph"]
        assert kg["meta"]["target"] == query

        nodes = kg["nodes"]
        assert "entity_ecosystem" in nodes
        assert "equity_analysis" in nodes
        assert "credit_analysis" in nodes
        assert "simulation_engine" in nodes
        assert "strategic_synthesis" in nodes

        # Check specific deep dive values (mocked)
        assert nodes["simulation_engine"]["monte_carlo_default_prob"] >= 0.0
        assert nodes["strategic_synthesis"]["final_verdict"]["conviction_level"] >= 1
