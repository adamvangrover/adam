# core/engine/crisis_simulation_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements the Crisis Simulation Graph (v23).
It is a specialized reasoning engine for macro-economic stress testing.
It uses a recursive approach:
1. Decompose the user scenario into macro variables (rates, GDP, inflation).
2. Simulate First-Order Impacts (direct hits to portfolio).
3. Simulate Second-Order Impacts (cascading failures, counterparty risk).
4. Critique the realism of the simulation.
5. Refine/Intensify if necessary.
"""

import logging
import random
from typing import Literal, Dict, Any, List
from core.utils.graph_utils import StateGraph, END, START, MemorySaver, HAS_LANGGRAPH
from core.engine.sector_impact_engine import SectorImpactEngine
from core.engine.states import CrisisSimulationState

logger = logging.getLogger(__name__)

# Initialize the engine
sector_engine = SectorImpactEngine()

# --- Mock Logic (To be replaced by CrisisSimulationPlugin / LLM) ---


def mock_decompose_scenario(scenario: str) -> Dict[str, float]:
    """Parses natural language scenario into quantitative shocks."""
    variables = {}
    scenario_lower = scenario.lower()

    if "rate" in scenario_lower:
        variables["interest_rate_bps"] = 50.0  # +50 bps
    if "recession" in scenario_lower or "gdp" in scenario_lower:
        variables["gdp_growth"] = -2.5  # -2.5%
    if "inflation" in scenario_lower:
        variables["cpi_inflation"] = 8.0  # 8%

    # Default if empty
    if not variables:
        variables["market_volatility_vix"] = 35.0

    return variables

def derive_sector_shocks(scenario: str) -> Dict[str, float]:
    """Maps scenario keywords to specific sector shocks."""
    shocks = {}
    s = scenario.lower()

    if "tech" in s or "cyber" in s or "ai" in s:
        shocks["Technology"] = -0.5
    if "energy" in s or "oil" in s:
        shocks["Energy"] = -0.4
    if "estate" in s or "housing" in s or "rate" in s:
        shocks["Real Estate"] = -0.6
        shocks["Financials"] = -0.3
    if "consumer" in s or "retail" in s:
        shocks["Consumer Discretionary"] = -0.4

    # Default shock if nothing specific found but it's a crisis
    if not shocks:
        shocks["Financials"] = -0.2

    return shocks

# --- Nodes ---


def decompose_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Decompose Scenario ---")
    vars = mock_decompose_scenario(state["scenario_description"])
    shocks = derive_sector_shocks(state["scenario_description"])

    return {
        "macro_variables": vars,
        "sector_shocks": shocks,
        "human_readable_status": f"Decomposed scenario into {len(vars)} macro variables and {len(shocks)} sector shocks."
    }


def simulate_direct_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Simulate Direct Impact ---")

    portfolio = state.get("portfolio_data", [])
    # Ensure portfolio is a list of dicts, if it's not, we might need to handle it.
    if isinstance(portfolio, dict) and "positions" in portfolio:
        portfolio = portfolio["positions"]
    elif not isinstance(portfolio, list):
        # Fallback for mock portfolio if empty or malformed
        portfolio = [{"name": "Generic Asset", "sector": "Financials", "leverage": 3.0, "rating": "BBB"}]

    shocks = state.get("sector_shocks", {})

    # Run the engine
    engine_result = sector_engine.analyze_portfolio(portfolio, custom_shocks=shocks)

    # Extract First Order Impacts (Direct Insights)
    impacts = []
    loss = 0.0
    for r in engine_result["results"]:
        # Only log significant negative impacts
        if "NEGATIVE" in r["macro_insight"] or "Critical" in r["credit_insight"]:
            impacts.append(f"{r['asset']}: {r['macro_insight']} | {r['credit_insight']}")

        # Simple loss model based on score (0-100 where 100 is bad)
        # Assume portfolio size of 100M per asset for simplicity
        risk_delta = r["consensus_score"] - 30.0 # Baseline risk approx 30
        if risk_delta > 0:
            loss += (risk_delta / 100.0) * 10.0 # $10M exposure

    # Extract Contagion Log for next step
    # engine_result["simulation_log"]["detailed_contagion_log"] is List[str]
    raw_log = engine_result["simulation_log"].get("detailed_contagion_log", [])

    # Format into CrisisLogEntry dicts
    structured_log = []
    for i, event in enumerate(raw_log):
        structured_log.append({
            "timestamp": f"T+{i+1}:00",
            "event_description": event,
            "risk_id_cited": "SYS-CONTAGION",
            "status": "Escalating"
        })

    return {
        "first_order_impacts": impacts,
        "estimated_loss": loss,
        "crisis_simulation_log": structured_log, # Pass to state
        "human_readable_status": f"Simulated direct impacts. Loss estimate: ${loss:.2f}M"
    }


def simulate_cascade_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Simulate Cascade ---")

    # Read the log from the previous step
    log_entries = state.get("crisis_simulation_log", [])
    cascades = [entry["event_description"] for entry in log_entries]

    # If no contagion, maybe add some generic ones if loss is high
    if not cascades and state["estimated_loss"] > 20.0:
         cascades.append("Liquidity crunch: Revolving credit lines drawn down.")

    # Intensify loss based on cascades
    additional_loss = len(cascades) * 1.5
    total_loss = state["estimated_loss"] + additional_loss

    return {
        "second_order_impacts": cascades,
        "estimated_loss": total_loss,
        "human_readable_status": f"Simulated cascading effects. Total Risk: ${total_loss:.2f}M"
    }


def critique_simulation_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Critique Simulation ---")
    iteration = state["iteration_count"]
    loss = state["estimated_loss"]

    critique_notes = []
    is_realistic = True
    needs_refinement = False

    # Critique Logic
    if loss < 5.0 and iteration < 2:
        critique_notes.append("Scenario seems too mild. Intensify shocks.")
        is_realistic = False
        needs_refinement = True

    if not state["second_order_impacts"] and iteration < 1:
        critique_notes.append("Missing second-order effects. Dig deeper.")
        is_realistic = False
        needs_refinement = True

    return {
        "critique_notes": critique_notes,
        "is_realistic": is_realistic,
        "needs_refinement": needs_refinement,
        "iteration_count": iteration + 1,
        "human_readable_status": "Critiqued simulation realism."
    }


def refine_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Refine/Intensify ---")
    # Intensify logic
    current_vars = state["macro_variables"]
    current_shocks = state.get("sector_shocks", {})

    new_vars = {k: v * 1.5 for k, v in current_vars.items()}
    new_shocks = {k: v * 1.5 for k, v in current_shocks.items()}

    return {
        "macro_variables": new_vars,
        "sector_shocks": new_shocks,
        "needs_refinement": False,  # Reset flag
        "human_readable_status": "Intensified scenario parameters."
    }


def generate_report_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Generate Crisis Report ---")
    report = f"Crisis Simulation Report: {state['scenario_description']}\n"
    report += "="*40 + "\n"
    report += f"Estimated Loss: ${state['estimated_loss']:.2f}M\n\n"

    report += "Direct Impacts:\n"
    for i in state["first_order_impacts"]:
        report += f"- {i}\n"

    report += "\nCascading Effects:\n"
    for i in state["second_order_impacts"]:
        report += f"- {i}\n"

    return {
        "final_report": report,
        "human_readable_status": "Report generated."
    }

# --- Conditional Logic ---


def should_continue_crisis(state: CrisisSimulationState) -> Literal["refine", "finalize"]:
    if state["needs_refinement"] and state["iteration_count"] < 3:
        return "refine"
    return "finalize"

# --- Graph Construction ---


def build_crisis_graph():
    if not HAS_LANGGRAPH:
        logger.warning("LangGraph not available. Crisis Graph will be mocked.")

    workflow = StateGraph(CrisisSimulationState)

    workflow.add_node("decompose", decompose_node)
    workflow.add_node("simulate_direct", simulate_direct_node)
    workflow.add_node("simulate_cascade", simulate_cascade_node)
    workflow.add_node("critique", critique_simulation_node)
    workflow.add_node("refine", refine_node)
    workflow.add_node("generate_report", generate_report_node)

    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "simulate_direct")
    workflow.add_edge("simulate_direct", "simulate_cascade")
    workflow.add_edge("simulate_cascade", "critique")

    workflow.add_conditional_edges(
        "critique",
        should_continue_crisis,
        {
            "refine": "refine",
            "finalize": "generate_report"
        }
    )

    workflow.add_edge("refine", "simulate_direct")  # Loop back to simulation
    workflow.add_edge("generate_report", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


crisis_simulation_app = build_crisis_graph()
