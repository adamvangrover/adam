# core/engine/esg_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements the ESG (Environmental, Social, Governance) Analysis Graph.
It evaluates a company's sustainability profile using a cyclical reasoning process.
It assesses individual E, S, and G factors, calculates an aggregate score,
and then critiques the findings against known controversies (greenwashing detection).
"""

import logging
from typing import Literal, Dict, Any
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

from core.engine.states import ESGAnalysisState

logger = logging.getLogger(__name__)

# --- Mock Utilities (In a real system, these would be in esg_utils.py) ---


def mock_analyze_env(company: str, sector: str) -> float:
    # Logic: Energy sector scores lower by default, Tech scores higher
    base_score = 70.0
    if sector.lower() in ["energy", "oil", "mining"]:
        base_score = 40.0
    elif sector.lower() in ["tech", "technology", "software"]:
        base_score = 85.0
    return base_score


def mock_analyze_social(company: str) -> float:
    return 75.0  # Average


def mock_analyze_gov(company: str) -> float:
    return 80.0  # Good governance assumed


def mock_check_controversies(company: str) -> list[str]:
    # Mock database lookup
    if "Oil" in company or "Energy" in company:
        return ["Spill incident in 2022", "Community protest regarding pipeline"]
    return []

# --- Nodes ---


def analyze_env_node(state: ESGAnalysisState) -> Dict[str, Any]:
    print("--- Node: Analyze Environmental ---")
    score = mock_analyze_env(state["company_name"], state["sector"])
    return {
        "env_score": score,
        "human_readable_status": f"Analyzed Environmental impact (Score: {score})."
    }


def analyze_social_node(state: ESGAnalysisState) -> Dict[str, Any]:
    print("--- Node: Analyze Social ---")
    score = mock_analyze_social(state["company_name"])
    return {
        "social_score": score,
        "human_readable_status": f"Analyzed Social impact (Score: {score})."
    }


def analyze_gov_node(state: ESGAnalysisState) -> Dict[str, Any]:
    print("--- Node: Analyze Governance ---")
    score = mock_analyze_gov(state["company_name"])
    return {
        "gov_score": score,
        "human_readable_status": f"Analyzed Governance structure (Score: {score})."
    }


def aggregate_esg_node(state: ESGAnalysisState) -> Dict[str, Any]:
    print("--- Node: Aggregate ESG Score ---")
    # Simple average
    total = (state["env_score"] + state["social_score"] + state["gov_score"]) / 3

    # Check controversies
    controversies = mock_check_controversies(state["company_name"])

    # Draft Report
    report = f"ESG Analysis for {state['company_name']}\n"
    report += f"Sector: {state['sector']}\n"
    report += f"Scores - E: {state['env_score']}, S: {state['social_score']}, G: {state['gov_score']}\n"
    report += f"Total Score: {total:.2f}\n"
    if controversies:
        report += f"Controversies Detected: {', '.join(controversies)}\n"
        # Penalty
        total -= (len(controversies) * 5)
        report += f"Adjusted Score (after controversies): {total:.2f}\n"

    return {
        "total_esg_score": total,
        "controversies": controversies,
        "final_report": report,
        "human_readable_status": f"Calculated Final ESG Score: {total:.2f}"
    }


def critique_esg_node(state: ESGAnalysisState) -> Dict[str, Any]:
    print("--- Node: Critique ESG ---")
    score = state["total_esg_score"]
    controversies = state["controversies"]
    iteration = state["iteration_count"]

    critique_notes = []
    needs_revision = False

    # Critique Logic
    if score > 80 and controversies:
        critique_notes.append("High score despite controversies. Verify severity of controversies.")
        needs_revision = True

    if iteration < 1:
        critique_notes.append("Perform deeper search for hidden subsidiaries (simulated).")
        needs_revision = True

    return {
        "critique_notes": critique_notes,
        "needs_revision": needs_revision,
        "iteration_count": iteration + 1,
        "human_readable_status": "Critiqued ESG findings."
    }


def revise_esg_node(state: ESGAnalysisState) -> Dict[str, Any]:
    print("--- Node: Revise ESG ---")
    report = state["final_report"]
    notes = state["critique_notes"]

    new_report = report + "\n[Revision Notes]\n"
    for note in notes:
        new_report += f"- {note}\n"

    return {
        "final_report": new_report,
        "needs_revision": False,
        "human_readable_status": "Revised ESG report."
    }

# --- Conditional Logic ---


def should_continue_esg(state: ESGAnalysisState) -> Literal["revise_esg", "END"]:
    if state["needs_revision"] and state["iteration_count"] < 3:
        return "revise_esg"
    return "END"

# --- Graph Construction ---


def build_esg_graph():
    if not HAS_LANGGRAPH:
        return None

    workflow = StateGraph(ESGAnalysisState)

    # Add Nodes
    workflow.add_node("analyze_env", analyze_env_node)
    workflow.add_node("analyze_social", analyze_social_node)
    workflow.add_node("analyze_gov", analyze_gov_node)
    workflow.add_node("aggregate_esg", aggregate_esg_node)
    workflow.add_node("critique_esg", critique_esg_node)
    workflow.add_node("revise_esg", revise_esg_node)

    # Edges
    # Parallel execution of E, S, G is simulated by sequential here,
    # but LangGraph allows parallel if we fan-out. Keeping it simple for now.
    workflow.add_edge(START, "analyze_env")
    workflow.add_edge("analyze_env", "analyze_social")
    workflow.add_edge("analyze_social", "analyze_gov")
    workflow.add_edge("analyze_gov", "aggregate_esg")
    workflow.add_edge("aggregate_esg", "critique_esg")

    workflow.add_conditional_edges(
        "critique_esg",
        should_continue_esg,
        {
            "revise_esg": "revise_esg",
            "END": END
        }
    )

    workflow.add_edge("revise_esg", "critique_esg")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


esg_graph_app = build_esg_graph()
