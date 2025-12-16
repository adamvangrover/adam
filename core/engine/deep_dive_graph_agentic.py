import logging
import asyncio
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

# v23 Imports
from core.engine.states import OmniscientState
from core.agents.specialized.snc_rating_agent import SNCRatingAgent
from core.agents.specialized.monte_carlo_risk_agent import MonteCarloRiskAgent
from core.agents.specialized.quantum_scenario_agent import QuantumScenarioAgent
from core.agents.specialized.covenant_analyst_agent import CovenantAnalystAgent

logger = logging.getLogger(__name__)

class DeepDiveGraph:
    """
    v23.5 'Deep Dive' Protocol.
    Orchestrates the 5-phase "Autonomous Financial Analyst" pipeline.

    Phases:
    1. Entity & Ecosystem (Pre-filled by Context)
    2. Deep Fundamental (Valuation)
    3. Credit & SNC (SNC Rating, Covenants)
    4. Risk & Simulation (Monte Carlo, Quantum)
    5. Synthesis (Strategic Conviction) - *Handled by Orchestrator's final pass*
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize Specialized Agents
        self.snc_agent = SNCRatingAgent(config)
        self.monte_carlo_agent = MonteCarloRiskAgent(config)
        self.quantum_agent = QuantumScenarioAgent(config)
        self.covenant_agent = CovenantAnalystAgent(config)

        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(OmniscientState)

        # Define Nodes
        workflow.add_node("snc_analysis", self._run_snc_analysis)
        workflow.add_node("covenant_analysis", self._run_covenant_analysis)
        workflow.add_node("monte_carlo_sim", self._run_monte_carlo)
        workflow.add_node("quantum_scenarios", self._run_quantum_scenarios)

        # Define Edges (Parallel Execution where possible)
        # We start with Credit Analysis (SNC + Covenant)
        workflow.set_entry_point("snc_analysis")

        # From SNC, fork to Covenant and Risk
        workflow.add_edge("snc_analysis", "covenant_analysis")
        workflow.add_edge("covenant_analysis", "monte_carlo_sim")
        workflow.add_edge("monte_carlo_sim", "quantum_scenarios")

        workflow.add_edge("quantum_scenarios", END)

        return workflow

    # --- Node Functions ---
    # Helper to access nested nodes safely
    def _get_nodes(self, state: OmniscientState) -> Dict[str, Any]:
        return state.get("v23_knowledge_graph", {}).get("nodes", {})

    async def _run_snc_analysis(self, state: OmniscientState) -> Dict[str, Any]:
        logger.info("--- Phase 3a: SNC Analysis ---")
        nodes = self._get_nodes(state)

        # Extract inputs (mocked here as they might come from Phase 2)
        # In a real run, we'd pull from nodes['equity_analysis']['fundamentals'] etc.
        # For now, we use placeholders or look for 'financials' if injected

        # Mock retrieval from existing state or injection
        financials = {
            "ebitda": 100.0,
            "total_debt": 350.0,
            "interest_expense": 20.0
        }
        cap_structure = [
            {"name": "Term Loan B", "amount": 250.0, "priority": 1},
            {"name": "Senior Notes", "amount": 100.0, "priority": 2}
        ]

        result = self.snc_agent.execute(
            financials=financials,
            capital_structure=cap_structure,
            enterprise_value=1000.0
        )

        # Update State
        # We need to return the FULL structure update for TypedDict if we want to be safe,
        # or rely on the fact that we are modifying the dict in place if passed by reference (risky in async).
        # LangGraph typically merges.

        # We assume 'credit_analysis' exists or we create it
        credit_analysis = nodes.get("credit_analysis", {})
        credit_analysis["snc_rating_model"] = result

        # We must return the structure that matches OmniscientState
        # Deep update approach:
        nodes["credit_analysis"] = credit_analysis

        return {
            "v23_knowledge_graph": {
                "meta": state["v23_knowledge_graph"]["meta"],
                "nodes": nodes
            },
            "human_readable_status": "SNC Analysis Complete."
        }

    async def _run_covenant_analysis(self, state: OmniscientState) -> Dict[str, Any]:
        logger.info("--- Phase 3b: Covenant Analysis ---")
        nodes = self._get_nodes(state)

        # Mock inputs
        current_leverage = 3.5

        result = await self.covenant_agent.execute(
            leverage=current_leverage,
            covenant_threshold=4.5
        )

        credit_analysis = nodes.get("credit_analysis", {})
        credit_analysis["covenant_risk_analysis"] = result
        nodes["credit_analysis"] = credit_analysis

        return {
            "v23_knowledge_graph": {
                "meta": state["v23_knowledge_graph"]["meta"],
                "nodes": nodes
            },
            "human_readable_status": "Covenant Analysis Complete."
        }

    async def _run_monte_carlo(self, state: OmniscientState) -> Dict[str, Any]:
        logger.info("--- Phase 4a: Monte Carlo Simulation ---")
        nodes = self._get_nodes(state)

        result = await self.monte_carlo_agent.execute(
            current_ebitda=100.0,
            ebitda_volatility=0.25,
            interest_expense=20.0,
            capex_maintenance=10.0
        )

        nodes["simulation_engine"] = result # SimulationEngine schema matches agent output

        return {
            "v23_knowledge_graph": {
                "meta": state["v23_knowledge_graph"]["meta"],
                "nodes": nodes
            },
            "human_readable_status": "Monte Carlo Simulation Complete."
        }

    async def _run_quantum_scenarios(self, state: OmniscientState) -> Dict[str, Any]:
        logger.info("--- Phase 4b: Quantum Scenarios ---")
        nodes = self._get_nodes(state)
        ticker = state.get("ticker", "Unknown")

        result = await self.quantum_agent.execute(
            ticker=ticker,
            financials={"total_assets": 1000, "total_debt": 350, "volatility": 0.25}
        )

        sim_engine = nodes.get("simulation_engine")
        # In Pydantic model 'SimulationEngine', 'quantum_scenarios' is a list.
        # 'result' is a list of QuantumScenario objects.
        if sim_engine:
             # If sim_engine is a Pydantic object (from previous step), we can use .quantum_scenarios
             # If it's a dict (from JSON serialization), we access via key.
             if hasattr(sim_engine, "quantum_scenarios"):
                 sim_engine.quantum_scenarios.extend(result)
             elif isinstance(sim_engine, dict):
                 if "quantum_scenarios" not in sim_engine:
                     sim_engine["quantum_scenarios"] = []
                 sim_engine["quantum_scenarios"].extend(result)

        nodes["simulation_engine"] = sim_engine

        return {
            "v23_knowledge_graph": {
                "meta": state["v23_knowledge_graph"]["meta"],
                "nodes": nodes
            },
            "human_readable_status": "Deep Dive Complete."
        }
