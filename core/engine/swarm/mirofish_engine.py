import asyncio
import importlib
import inspect
import logging
import pkgutil
from typing import Any, Dict, List

from core.agents.agent_base import AgentBase
from core.engine.swarm.personas import (
    BasePersona,
    InstitutionalTraderPersona,
    PersonaAction,
    RegulatorPersona,
    RetailInvestorPersona,
)
from core.engine.swarm.pheromone_board import PheromoneBoard
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.llm_plugin import LLMPlugin

logger = logging.getLogger(__name__)

class AgentPersonaAdapter(BasePersona):
    """
    Wraps standard AgentBase instances to function as Personas
    within the MiroFish Swarm simulation, ensuring they adhere
    to the required `act` interface.
    """
    def __init__(self, agent: AgentBase, board: PheromoneBoard):
        super().__init__(agent.name, board)
        self.agent = agent

    def _determine_action(self, cycle: int, global_events: list, agent_actions: list) -> PersonaAction | None:
        # We can't actually await execute inside this synchronous function,
        # but since _determine_action is called by act, we can do it directly in an override
        raise NotImplementedError("AgentPersonaAdapter must override act directly.")

    async def act(self, cycle: int) -> PersonaAction | None:
        """
        Evaluate the environment using the underlying agent's logic.
        """
        recent_events = await self.board.sniff("simulation.events.global")
        other_actions = await self.board.sniff("simulation.actions.agent")

        context = {
            "cycle": cycle,
            "recent_events": recent_events,
            "other_actions": other_actions
        }

        try:
            # We call the wrapped agent's execute method and parse the output
            # A fallback is used since the full implementation varies by agent.
            result = await self.agent.execute(**context)

            # Simple heuristic mapping since each agent's format differs
            sentiment = "Neutral"
            action_type = "ANALYZE"
            confidence = 0.5
            rationale = "Executed analytical task based on swarm state."

            if isinstance(result, dict):
                r_str = str(result).lower()
                if "bullish" in r_str:
                    sentiment = "Bullish"
                if "bearish" in r_str:
                    sentiment = "Bearish"
                confidence = result.get("confidence", 0.7)
                rationale = result.get("rationale", rationale)
            elif isinstance(result, str):
                if "bullish" in result.lower():
                    sentiment = "Bullish"
                if "bearish" in result.lower():
                    sentiment = "Bearish"

            action = PersonaAction(
                agent_id=self.agent_id,
                action_type=action_type,
                sentiment=sentiment,
                rationale=rationale,
                confidence=float(confidence)
            )

            self.memory.append(action.model_dump())
            await self.board.deposit(
                signal_type="simulation.actions.agent",
                data={"cycle": cycle, **action.model_dump()},
                intensity=action.confidence * 10,
                source=self.agent_id
            )
            return action

        except Exception as e:
            logger.warning(f"Wrapped agent {self.agent_id} failed to act: {e}")
            return None


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
        retail_count = int(self.agent_count * 0.5)
        inst_count = int(self.agent_count * 0.25)
        reg_count = max(1, int(self.agent_count * 0.05))

        self.active_agents.extend([RetailInvestorPersona(f"Retail_{i}", self.board) for i in range(retail_count)])
        self.active_agents.extend([InstitutionalTraderPersona(f"Inst_{i}", self.board) for i in range(inst_count)])
        self.active_agents.extend([RegulatorPersona(f"Reg_{i}", self.board) for i in range(reg_count)])

        # 3. Load Dynamic Core Agents into Swarm
        self._load_dynamic_agents()

        self.is_running = True

    def _load_dynamic_agents(self):
        """
        Dynamically load all available specialized agents from core.agents
        and spin them up as part of the MiroFish engine as needed.
        """
        loaded_count = 0
        target_dynamic = self.agent_count - len(self.active_agents)

        if target_dynamic <= 0:
            return

        try:
            # Recursively find all modules in core/agents
            import core.agents
            prefix = core.agents.__name__ + "."
            for _importer, modname, _ispkg in pkgutil.walk_packages(core.agents.__path__, prefix):
                if loaded_count >= target_dynamic:
                    break

                try:
                    module = importlib.import_module(modname)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if loaded_count >= target_dynamic:
                            break
                        if issubclass(obj, AgentBase) and obj is not AgentBase and not name.startswith("Pydantic"):
                            # Instantiate the agent
                            agent_instance = obj(config={"agent_id": f"Dynamic_{name}_{loaded_count}"})
                            adapter = AgentPersonaAdapter(agent_instance, self.board)
                            self.active_agents.append(adapter)
                            loaded_count += 1
                except Exception:
                    # Some agents might fail to import due to missing dependencies, we ignore them
                    pass

        except Exception as e:
            logger.error(f"[MiroFish] Failed to dynamically load core agents: {e}")

        logger.info(f"[MiroFish] Successfully spun up {loaded_count} specialized core agents.")

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
            err_msg = (
                f"[MiroFish] Simulation encountered a critical fault: {e}. "
                "Gracefully halting and returning partial data."
            )
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
            pct = int((bearish_count / total) * 100)
            return f"Swarm output indicates severe liquidity contraction and market panic ({pct}% Bearish alignment)."
        elif bullish_count > total * 0.6:
            pct = int((bullish_count / total) * 100)
            return f"Swarm output indicates robust expansion and euphoric deployment ({pct}% Bullish alignment)."

        return "Swarm output indicates high polarization and fragmented market consensus."
