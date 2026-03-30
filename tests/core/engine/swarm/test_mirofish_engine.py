import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from core.engine.swarm.mirofish_engine import MiroFishSwarmEngine
from core.engine.swarm.personas import RetailInvestorPersona, InstitutionalTraderPersona, RegulatorPersona
from core.engine.swarm.pheromone_board import PheromoneBoard

@pytest.mark.asyncio
async def test_mirofish_engine_wind_up_and_wind_down():
    """
    Tests the Wind-Up (agent spawning) and Graceful Wind-Down (clean up)
    logic in the MiroFish engine as mandated by the ARCHITECT_INFINITE protocol.
    """
    engine = MiroFishSwarmEngine(agent_count=10, max_cycles=2, consensus_threshold=0.8)

    # Test Wind-Up
    await engine.initialize_environment({"target": "NVDA_Earnings"})

    # Check if heterogeneous agents were properly spawned based on percentages
    assert len(engine.active_agents) == 10
    retail_count = sum(1 for a in engine.active_agents if isinstance(a, RetailInvestorPersona))
    inst_count = sum(1 for a in engine.active_agents if isinstance(a, InstitutionalTraderPersona))
    reg_count = sum(1 for a in engine.active_agents if isinstance(a, RegulatorPersona))

    assert retail_count == 6  # 60%
    assert inst_count == 3    # 30%
    assert reg_count == 1     # 10%
    assert engine.is_running is True

    # Test execution & Wind-Down
    results = await engine.run_simulation_cycles(cycles=1)

    assert engine.is_running is False
    assert len(engine.active_agents) == 0
    assert isinstance(results, list)

@pytest.mark.asyncio
async def test_mirofish_engine_early_consensus_termination():
    """
    Tests whether the engine correctly halts execution early if the
    swarm achieves consensus, saving computational budgets.
    """
    engine = MiroFishSwarmEngine(agent_count=5, max_cycles=5, consensus_threshold=0.6)
    await engine.initialize_environment({"target": "Liquidity_Crisis"})

    # Mock the board's sniff method to immediately return a strong bearish consensus
    async def mock_sniff(signal_type):
        from core.engine.swarm.pheromone_board import Pheromone
        # Inject 4 bearish signals out of 5 for cycle 0 (80% > 60% threshold)
        return [
            Pheromone(signal_type="simulation.actions.agent", data={"cycle": 0, "sentiment": "Bearish"}, intensity=10, source_agent_id="a1"),
            Pheromone(signal_type="simulation.actions.agent", data={"cycle": 0, "sentiment": "Bearish"}, intensity=10, source_agent_id="a2"),
            Pheromone(signal_type="simulation.actions.agent", data={"cycle": 0, "sentiment": "Bearish"}, intensity=10, source_agent_id="a3"),
            Pheromone(signal_type="simulation.actions.agent", data={"cycle": 0, "sentiment": "Bearish"}, intensity=10, source_agent_id="a4"),
            Pheromone(signal_type="simulation.actions.agent", data={"cycle": 0, "sentiment": "Bullish"}, intensity=10, source_agent_id="a5")
        ]

    engine.board.sniff = AsyncMock(side_effect=mock_sniff)

    # Only cycle 0 should run; it breaks immediately after checking consensus
    with patch('core.engine.swarm.mirofish_engine.logger.info') as mock_logger:
        await engine.run_simulation_cycles(cycles=5)

        # Verify the early consensus log was triggered
        early_consensus_logs = [call_args[0][0] for call_args in mock_logger.call_args_list if "Early consensus reached" in call_args[0][0]]
        assert len(early_consensus_logs) == 1
        assert engine.is_running is False

@pytest.mark.asyncio
async def test_mirofish_engine_graceful_degradation():
    """
    Tests that a critical failure during simulation execution
    is gracefully caught, logged, and the environment is cleaned up.
    """
    engine = MiroFishSwarmEngine(agent_count=5, max_cycles=2)
    await engine.initialize_environment({"target": "Stress_Test"})

    # Intentionally corrupt the active agents list to cause an exception during run
    engine.active_agents = None

    with patch('core.engine.swarm.mirofish_engine.logger.error') as mock_logger:
        # Should not raise exception
        results = await engine.run_simulation_cycles(cycles=2)

        # Verify graceful degradation log was triggered
        error_logs = [call_args[0][0] for call_args in mock_logger.call_args_list if "critical fault" in call_args[0][0]]
        assert len(error_logs) > 0

        # Ensure environment still wound down
        assert engine.is_running is False
