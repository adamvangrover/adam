from unittest.mock import patch

import pytest

from core.engine.meta_orchestrator import MetaOrchestrator


@pytest.mark.asyncio
async def test_code_gen_pipeline():
    # Mock LLMPlugin to avoid API key requirements and AgentOrchestrator initialization issues
    with patch('core.system.agent_orchestrator.LLMPlugin') as MockLLM:
        orchestrator = MetaOrchestrator()

        # Trigger Code Gen
        query = "Generate code to calculate fibonacci sequence in python"

        # Verify complexity assessment
        complexity = orchestrator._assess_complexity(query)
        assert complexity == "CODE_GEN"

        # Execute
        result = await orchestrator.route_request(query)

        assert result is not None
        assert "result" in result
        assert result["result"]["status"] == "Code Generated"
        assert "def execute_task" in result["result"]["code"]
        assert "calculate fibonacci" in result["result"]["code"]
