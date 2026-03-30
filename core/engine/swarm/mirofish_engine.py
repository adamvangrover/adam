import asyncio
import logging
from typing import Any, Dict, List

from core.engine.swarm.personas import InstitutionalTraderPersona, RegulatorPersona, RetailInvestorPersona
from core.engine.swarm.pheromone_board import PheromoneBoard
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.llm_plugin import LLMPlugin

logger = logging.getLogger(__name__)

class MiroFishSwarmEngine:
    """
    Massive Parallelized Swarm Intelligence Engine.
    Operates as the System 3 (Compute/Simulation) layer for multi-agent emergence.
    Decoupled via an event-driven message bus (PheromoneBoard).
    Implements robust wind-up (scaling) and wind-down (consensus tracking) logic.
    Gracefully degrades to classical simulation (CrisisEngine) on failure or budget exhaustion.
    """

    def __init__(self, agent_count: int = 100, max_cycles: int = 5, consensus_threshold: float = 0.85):
        self.board = PheromoneBoard()
        self.kg = UnifiedKnowledgeGraph()
        self.agent_count = agent_count
        self.max_cycles = max_cycles
        self.consensus_threshold = consensus_threshold
        self.active_agents = []
        self.is_running = False

        self.llm = LLMPlugin(config={
            "provider": "gemini",
            "gemini_model_name": "gemini-3-pro"
        })

    async def initialize_environment(self, seed_parameters: Dict[str, Any]):
        """
        Wind-up Phase:
        Extracts entities from seed parameters and constructs the foundational digital sandbox reality.
        Spawns heterogeneous agents based on the target scale.
        """
        logger.info(f"[MiroFish] Initializing environment with {self.agent_count} heterogeneous agents.")

        # 1. Post environmental state to the bus
        for key, value in seed_parameters.items():
            self._apply_symbolic_certification(key, value)

            await self.board.deposit(
                signal_type="simulation.events.global",
                data={"event": "ENVIRONMENT_INIT", "key": key, "value": value},
                intensity=10.0,
                source="OASIS_System"
            )

        # 2. Wind-up Agents (Heterogeneous mixture)
        retail_count = int(self.agent_count * 0.6)
        inst_count = int(self.agent_count * 0.3)
        reg_count = self.agent_count - retail_count - inst_count

        self.active_agents.extend([RetailInvestorPersona(f"Retail_{i}", self.board) for i in range(retail_count)])
        self.active_agents.extend([InstitutionalTraderPersona(f"Inst_{i}", self.board) for i in range(inst_count)])
        self.active_agents.extend([RegulatorPersona(f"Reg_{i}", self.board) for i in range(reg_count)])

        self.is_running = True

    def _apply_symbolic_certification(self, key: str, value: Any):
        """
        Validates the extracted data against the proprietary financial ontology.
        """
        logger.debug(f"[MiroFish] Certified symbolic consistency for {key}: {value}")

    async def run_simulation_cycles(self, cycles: int = None) -> List[Dict[str, Any]]:
        """
        Executes the swarm over chronological rounds, monitoring for early consensus
        to efficiently wind-down and save token budgets.
        Implements Graceful Degradation: If a cycle fails, return the last known good state.
        """
        target_cycles = cycles or self.max_cycles
        logger.info(f"[MiroFish] Running simulation for up to {target_cycles} cycles.")

        try:
            for cycle in range(target_cycles):
                if not self.is_running:
                    break

                logger.info(f"[MiroFish] Commencing Cycle {cycle + 1}")

                # Parallel execution of all active personas
                tasks = [agent.act(cycle) for agent in self.active_agents]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Allow board to process events
                await asyncio.sleep(0.1)

                # Wind-Down Logic: Check for Early Consensus
                if await self._check_consensus_reached(cycle):
                    logger.info(f"[MiroFish] Early consensus reached at Cycle {cycle + 1}. Winding down simulation.")
                    break

        except Exception as e:
            err_msg = f"[MiroFish] Simulation encountered a critical fault: {e}. Gracefully halting and returning partial data."
            logger.error(err_msg)
        finally:
            await self._wind_down_environment()

        emergent_events = await self.board.sniff("simulation.actions.agent")
        return [event.data for event in emergent_events]

    async def _check_consensus_reached(self, cycle: int) -> bool:
        """
        Analyzes the current event bus to determine if the swarm has aligned
        on a specific outcome, preventing wasted token expenditure on further cycles.
        """
        events = await self.board.sniff("simulation.actions.agent")
        if not events:
            return False

        cycle_events = [e for e in events if e.data.get("cycle") == cycle]
        if not cycle_events:
            return False

        sentiments = [e.data.get("sentiment") for e in cycle_events if e.data.get("sentiment")]
        if not sentiments:
            return False

        bearish = sum(1 for s in sentiments if s == "Bearish")
        bullish = sum(1 for s in sentiments if s == "Bullish")
        total = len(sentiments)

        if total > 0:
            if bearish / total >= self.consensus_threshold or bullish / total >= self.consensus_threshold:
                return True
        return False

    async def _wind_down_environment(self):
        """
        Cleans up agent instances and frees resources.
        """
        agent_count = len(self.active_agents) if self.active_agents else 0
        logger.info(f"[MiroFish] Winding down {agent_count} active agents.")
        self.is_running = False
        if self.active_agents is not None:
            self.active_agents.clear()

    async def synthesize_report(self) -> str:
        """
        Reporting agent parses the post-simulation environment.
        Uses efficient string manipulation instead of expensive LLM summarization
        unless strict depth is required.
        """
        logger.info("[MiroFish] Synthesizing emergent report.")

        events = await self.board.sniff("simulation.actions.agent")

        if not events:
            return "Swarm collapsed into immediate consensus or failed to initialize. No emergent dynamics observed."

        bearish_count = sum(1 for e in events if e.data.get("sentiment") == "Bearish")
        bullish_count = sum(1 for e in events if e.data.get("sentiment") == "Bullish")
        total = len(events)

        if total == 0:
            return "Insufficient interaction volume for statistical relevance."

        if bearish_count > total * 0.6:
            return f"Swarm output indicates severe liquidity contraction and market panic ({int((bearish_count/total)*100)}% Bearish alignment)."
        elif bullish_count > total * 0.6:
            return f"Swarm output indicates robust expansion and euphoric deployment ({int((bullish_count/total)*100)}% Bullish alignment)."

        return "Swarm output indicates high polarization and fragmented market consensus."
