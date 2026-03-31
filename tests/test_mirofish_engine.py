import asyncio
import pytest
from typing import Any, Dict

from core.engine.swarm.mirofish_engine import MiroFishSwarmEngine, AgentPersonaAdapter
from core.engine.swarm.personas import RetailInvestorPersona, InstitutionalTraderPersona, RegulatorPersona
from core.agents.agent_base import AgentBase
from core.engine.meta_orchestrator import MetaOrchestrator


class MockAgent(AgentBase):
    async def execute(self, *args, **kwargs) -> Any:
        return {"sentiment": "Bullish", "confidence": 0.9, "rationale": "mock analysis"}

    def get_skill_schema(self) -> Dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_initialize_environment_with_dynamic_agents():
    engine = MiroFishSwarmEngine(agent_count=10)
    await engine.initialize_environment(seed_parameters={"target": "AAPL"})

    assert engine.is_running is True
    assert len(engine.active_agents) == 10

    # Verify heterogeneous mix
    retail = [a for a in engine.active_agents if isinstance(a, RetailInvestorPersona)]
    inst = [a for a in engine.active_agents if isinstance(a, InstitutionalTraderPersona)]
    reg = [a for a in engine.active_agents if isinstance(a, RegulatorPersona)]
    dynamic = [a for a in engine.active_agents if isinstance(a, AgentPersonaAdapter)]

    # 50% retail, 25% inst, 5% reg (min 1) => 5 retail, 2 inst, 1 reg
    # This leaves 2 spots for dynamic agents
    assert len(retail) == 5
    assert len(inst) == 2
    assert len(reg) >= 1
    assert len(dynamic) > 0


@pytest.mark.asyncio
async def test_run_simulation_cycles_max():
    engine = MiroFishSwarmEngine(agent_count=2, max_cycles=3, consensus_threshold=1.1)
    await engine.initialize_environment({"target": "AAPL"})

    # Clear active agents and just add one mock that doesn't trigger consensus
    class NeutralAgent(AgentPersonaAdapter):
        async def act(self, cycle):
            await asyncio.sleep(0.01)
            return None

    mock_agent = MockAgent(config={"agent_id": "test_agent"})
    engine.active_agents = [NeutralAgent(mock_agent, engine.board)]

    result = await engine.run_simulation_cycles(cycles=3)
    assert engine.is_running is False


@pytest.mark.asyncio
async def test_run_simulation_cycles_early_consensus():
    engine = MiroFishSwarmEngine(agent_count=2, max_cycles=5, consensus_threshold=0.5)
    await engine.initialize_environment({"target": "AAPL"})

    # Force consensus immediately
    class ForceBullishAgent(AgentPersonaAdapter):
        async def act(self, cycle):
            await self.board.deposit(
                signal_type="simulation.actions.agent",
                data={"cycle": cycle, "sentiment": "Bullish"},
                intensity=10.0,
                source=self.agent_id
            )
            return None

    mock_agent = MockAgent(config={"agent_id": "test_agent"})
    engine.active_agents = [ForceBullishAgent(mock_agent, engine.board)]

    result = await engine.run_simulation_cycles(cycles=5)
    # Should stop early
    assert engine.is_running is False


@pytest.mark.asyncio
async def test_synthesize_report():
    engine = MiroFishSwarmEngine(agent_count=2)
    # No events
    assert "failed to initialize" in await engine.synthesize_report()

    # Bearish
    await engine.board.deposit("simulation.actions.agent", {"sentiment": "Bearish"}, 10.0, "1")
    await engine.board.deposit("simulation.actions.agent", {"sentiment": "Bearish"}, 10.0, "2")
    report = await engine.synthesize_report()
    assert "liquidity contraction" in report

    # Bullish
    engine.board._pheromones.clear()
    await engine.board.deposit("simulation.actions.agent", {"sentiment": "Bullish"}, 10.0, "1")
    await engine.board.deposit("simulation.actions.agent", {"sentiment": "Bullish"}, 10.0, "2")
    report = await engine.synthesize_report()
    assert "euphoric deployment" in report


@pytest.mark.asyncio
async def test_personas_logic():
    engine = MiroFishSwarmEngine(agent_count=1)

    # Retail
    retail = RetailInvestorPersona("r1", engine.board)
    action = await retail.act(1)
    assert action is not None
    assert action.action_type in ["PANIC_SELL", "FOMO_BUY", "POST"]

    # Inst
    inst = InstitutionalTraderPersona("i1", engine.board)
    action = await inst.act(1)
    assert action is not None
    assert action.action_type in ["LIQUIDITY_PROVISION", "SHORT_SALE", "HOLD"]

    # Regulator
    reg = RegulatorPersona("reg1", engine.board)
    # Only acts on stress or randomly
    action = await reg.act(1)
    if action:
        assert action.action_type in ["INTERVENTION", "OBSERVE"]


@pytest.mark.asyncio
async def test_mirofish_engine_graceful_degradation():
    orchestrator = MetaOrchestrator()
    # Mock initialize_environment to raise an exception
    async def mock_init(*args, **kwargs):
        raise Exception("Simulated Failure")

    orchestrator.mirofish_engine.initialize_environment = mock_init

    result = await orchestrator._run_swarm_flow("test query")
    # Gracefully falls back to CrisisSimulationEngine which returns "Crisis Simulation Complete"
    assert result.get("status", "") == "Crisis Simulation Complete" or "error" in result
