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
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    class StateGraph:
         def __init__(self, *args, **kwargs): pass
         def add_node(self, *args, **kwargs): pass
         def add_edge(self, *args, **kwargs): pass
         def set_entry_point(self, *args, **kwargs): pass
         def add_conditional_edges(self, *args, **kwargs): pass
         def compile(self, *args, **kwargs): return None
    END = "END"
    START = "START"
    class MemorySaver: pass
    logger.warning("LangGraph not installed. Graphs will be disabled.")

from core.engine.states import CrisisSimulationState

logger = logging.getLogger(__name__)

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


def mock_simulate_impact(portfolio: Dict, macro_vars: Dict) -> tuple[List[str], float]:
    """Calculates direct hits."""
    impacts = []
    loss = 0.0

    if "interest_rate_bps" in macro_vars:
        impacts.append("Cost of debt increases for leveraged assets.")
        loss += 1.5  # Mock $M
    if "gdp_growth" in macro_vars:
        impacts.append("Revenue forecast downgrade across cyclical sectors.")
        loss += 3.2

    return impacts, loss


def mock_simulate_cascade(first_order: List[str]) -> List[str]:
    """Calculates knock-on effects."""
    cascades = []
    if any("debt" in i for i in first_order):
        cascades.append("Liquidity crunch: Revolving credit lines drawn down.")
        cascades.append("Covenant breach risk for Entity A.")
    if any("Revenue" in i for i in first_order):
        cascades.append("Supply chain partners delaying payments (DSO increase).")

    return cascades

# --- Nodes ---


def decompose_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Decompose Scenario ---")
    vars = mock_decompose_scenario(state["scenario_description"])
    return {
        "macro_variables": vars,
        "human_readable_status": f"Decomposed scenario into {len(vars)} macro variables."
    }


def simulate_direct_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Simulate Direct Impact ---")
    impacts, loss = mock_simulate_impact(state["portfolio_data"], state["macro_variables"])
    return {
        "first_order_impacts": impacts,
        "estimated_loss": loss,
        "human_readable_status": f"Simulated direct impacts. Loss estimate: ${loss}M"
    }


def simulate_cascade_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Simulate Cascade ---")
    cascades = mock_simulate_cascade(state["first_order_impacts"])

    # Intensify loss based on cascades
    additional_loss = len(cascades) * 0.5
    total_loss = state["estimated_loss"] + additional_loss

    return {
        "second_order_impacts": cascades,
        "estimated_loss": total_loss,
        "human_readable_status": f"Simulated cascading effects. Total Risk: ${total_loss}M"
    }


def critique_simulation_node(state: CrisisSimulationState) -> Dict[str, Any]:
    print("--- Node: Critique Simulation ---")
    iteration = state["iteration_count"]
    loss = state["estimated_loss"]

    critique_notes = []
    is_realistic = True
    needs_refinement = False

    # Critique Logic
    if loss < 1.0 and iteration < 2:
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
    new_vars = current_vars.copy()

    for k, v in new_vars.items():
        new_vars[k] = v * 1.5  # Increase shock by 50%

    return {
        "macro_variables": new_vars,
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
        return None

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
