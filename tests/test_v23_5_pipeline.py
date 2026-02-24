import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.engine.meta_orchestrator import MetaOrchestrator


@pytest.mark.integration
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
        # The mock logic in MetaOrchestrator defaults to AAPL if not specified
        # or uses the query. Let's check what it actually does.
        # In _run_deep_dive_flow: ticker = "AAPL" by default.
        # init_omniscient_state uses this ticker as the target.
        # Fallback uses query or extracted name.
        target = kg["meta"]["target"]
        assert target in ["AAPL", "TestCorp", query, "MockString"]

        nodes = kg["nodes"]
        assert "entity_ecosystem" in nodes
        assert "equity_analysis" in nodes
        assert "credit_analysis" in nodes
        assert "simulation_engine" in nodes
        assert "strategic_synthesis" in nodes

        # Check specific deep dive values (mocked)
        assert isinstance(nodes["simulation_engine"]["monte_carlo_default_prob"], str)
        assert nodes["strategic_synthesis"]["final_verdict"]["conviction_level"] >= 1
