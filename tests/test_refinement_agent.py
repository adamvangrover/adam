import pytest
from core.agents.refinement_agent import RefinementAgent
from core.agents.agent_base import AgentInput

@pytest.mark.asyncio
async def test_refinement_agent_execute():
    agent = RefinementAgent(config={"max_iterations": 1})
    input_data = AgentInput(query="Write a story about a dog.")

    output = await agent.execute(input_data)

    assert output.answer is not None
    assert output.metadata["iterations"] > 0
    assert "log_path" in output.metadata
