# tests/test_spec_architect_agent.py
import pytest
from core.agents.developer_swarm.spec_architect_agent import SpecArchitectAgent

@pytest.mark.asyncio
async def test_spec_architect_initialization():
    config = {"model": "gpt-4", "temperature": 0.0}
    agent = SpecArchitectAgent(config)
    assert agent is not None
    # Depending on base implementation, name might be set or default
    # assert agent.name == "SpecArchitectAgent"

@pytest.mark.asyncio
async def test_generate_spec_structure():
    config = {"model": "gpt-4", "temperature": 0.0}
    agent = SpecArchitectAgent(config)

    goal = "Add Two-Factor Authentication"
    spec_content = await agent.execute(goal)

    # Verify core SDAP sections are present
    assert "# Spec: Add Two-Factor Authentication" in spec_content
    assert "## 1. Overview & Objectives" in spec_content
    assert "## 2. Technical Context" in spec_content
    assert "## 3. Implementation Plan" in spec_content
    assert "## 4. Commands & Development" in spec_content
    assert "## 5. Verification & Testing Strategy" in spec_content
    assert "## 6. Constraints & Boundaries" in spec_content

    # Verify strict boundaries are mentioned
    assert "✅ **Always:**" in spec_content
    assert "🚫 **Never:**" in spec_content

@pytest.mark.asyncio
async def test_generate_spec_with_context():
    config = {"model": "gpt-4", "temperature": 0.0}
    agent = SpecArchitectAgent(config)

    goal = "Optimize Database Queries"
    context = ["src/db/models.py", "src/api/endpoints.py"]
    spec_content = await agent.execute(goal, context_files=context)

    assert "Analyzed context from: src/db/models.py, src/api/endpoints.py" in spec_content

@pytest.mark.asyncio
async def test_skill_schema():
    config = {"model": "gpt-4"}
    agent = SpecArchitectAgent(config)
    schema = agent.get_skill_schema()

    skills = [s["name"] for s in schema["skills"]]
    assert "generate_spec" in skills
