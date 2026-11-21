import pytest
import asyncio
from core.v30_architecture.python_intelligence.agents.sovereign_orchestrator import SovereignOrchestrator
from core.v30_architecture.python_intelligence.agents.adversarial_red_team import AdversarialRedTeamAgent
from core.v30_architecture.python_intelligence.agents.hardened_shield import HardenedShieldAgent

@pytest.mark.asyncio
async def test_sovereign_orchestrator_initialization():
    agent = SovereignOrchestrator()
    assert agent.name == "SovereignOrchestrator"
    assert agent.role == "orchestration"

@pytest.mark.asyncio
async def test_adversarial_red_team_initialization():
    agent = AdversarialRedTeamAgent()
    assert agent.name == "AdversarialRedTeam"
    assert agent.role == "security_red_team"

@pytest.mark.asyncio
async def test_hardened_shield_initialization():
    agent = HardenedShieldAgent()
    assert agent.name == "HardenedShield"
    assert agent.role == "security_defense"

# The agents' logic consists of infinite loops emitting packets.
# Full integration tests would require mocking the NeuralMesh.
# We just verify instantiation and base properties here.
