# core/engine/snc_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements a specialized Cyclical Reasoning Graph for Shared National Credits (SNC).
It orchestrates the analysis of complex syndicated loans, determining regulatory ratings
(Pass, Special Mention, Substandard, etc.) through an iterative critique-correction loop.
It leverages the v23 LangGraph architecture.
"""

import logging
from typing import Literal, Dict, Any
from core.utils.graph_utils import StateGraph, END, START, MemorySaver, HAS_LANGGRAPH
from core.engine.states import SNCAnalysisState
from core.engine.snc_utils import (
    calculate_leverage,
    map_financials_to_rating,
    analyze_syndicate_structure
)

logger = logging.getLogger(__name__)

# --- Nodes ---


def analyze_structure_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Analyze Structure
    Evaluates the syndicate composition and structural risks.
    """
    print("--- Node: Analyze Structure (SNC) ---")
    syndicate = state["syndicate_data"]

    analysis = analyze_syndicate_structure(syndicate)

    return {
        "structure_analysis": analysis,
        "human_readable_status": "Analyzed syndicate structure."
    }


def assess_credit_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Assess Credit
    Calculates metrics and drafts the initial regulatory rating.
    """
    print("--- Node: Assess Credit (SNC) ---")
    fin = state["financials"]

    # 1. Calculate Metrics
    ebitda = fin.get("ebitda", 0)
    debt = fin.get("total_debt", 0)
    liquidity = fin.get("liquidity", 0)

    leverage = calculate_leverage(debt, ebitda)

    # 2. Determine Baseline Rating
    rating = map_financials_to_rating(leverage, liquidity, debt)

    # 3. Draft Rationale
    rationale = f"Initial Assessment: {rating}\n"
    rationale += f"Metrics: Leverage {leverage:.2f}x, Liquidity ${liquidity}M.\n"
    if state["structure_analysis"]:
        rationale += f"Structural Notes: {state['structure_analysis']}\n"

    return {
        "regulatory_rating": rating,
        "rationale": rationale,
        "human_readable_status": f"Drafted rating: {rating}."
    }


def critique_snc_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Critique SNC Analysis
    Reviews the rating for regulatory compliance and logical consistency.
    """
    print("--- Node: Critique SNC ---")
    rating = state["regulatory_rating"]
    rationale = state["rationale"]
    iteration = state["iteration_count"]

    critique_notes = []
    needs_revision = False

    # Logic: Check for contradictions or missing context
    if rating == "Pass" and "Concentration Risk" in (state["structure_analysis"] or ""):
        critique_notes.append(
            "Rating is Pass, but structural concentration risk was noted. Justify why this doesn't warrant Special Mention.")
        needs_revision = True

    if iteration < 1:
        # Force at least one refinement pass to "simulate" deep thought
        critique_notes.append("Review leverage against industry peers (simulated).")
        needs_revision = True

    return {
        "critique_notes": critique_notes,
        "needs_revision": needs_revision,
        "iteration_count": iteration + 1,
        "human_readable_status": "Critiqued analysis."
    }


def revise_snc_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Revise Analysis
    Updates the rationale based on critique notes.
    """
    print("--- Node: Revise SNC ---")
    rationale = state["rationale"]
    notes = state["critique_notes"]

    # In a real LLM system, we would rewrite. Here we append the resolution.
    new_rationale = rationale + "\n\n[Revision]\n"
    for note in notes:
        new_rationale += f"- Addressed: {note}\n"

    return {
        "rationale": new_rationale,
        "needs_revision": False,  # Assume fixed
        "human_readable_status": "Revised rationale."
    }


def human_approval_node(state: SNCAnalysisState) -> Dict[str, Any]:
    print("--- Node: Human Approval ---")
    return {"human_readable_status": "Awaiting Final Sign-off."}

# --- Conditional Logic ---


def should_continue_snc(state: SNCAnalysisState) -> Literal["revise_snc", "human_approval", "END"]:
    if state["needs_revision"] and state["iteration_count"] < 3:
        return "revise_snc"

    # If rating is adverse, maybe require human approval?
    if state["regulatory_rating"] in ["Substandard", "Doubtful", "Loss"]:
        return "human_approval"

    return "END"

# --- Graph Construction ---


def build_snc_graph():
    if not HAS_LANGGRAPH:
        logger.warning("LangGraph not available. SNC Graph will be mocked.")

    workflow = StateGraph(SNCAnalysisState)

    workflow.add_node("analyze_structure", analyze_structure_node)
    workflow.add_node("assess_credit", assess_credit_node)
    workflow.add_node("critique_snc", critique_snc_node)
    workflow.add_node("revise_snc", revise_snc_node)
    workflow.add_node("human_approval", human_approval_node)

    workflow.add_edge(START, "analyze_structure")
    workflow.add_edge("analyze_structure", "assess_credit")
    workflow.add_edge("assess_credit", "critique_snc")

    workflow.add_conditional_edges(
        "critique_snc",
        should_continue_snc,
        {
            "revise_snc": "revise_snc",
            "human_approval": "human_approval",
            "END": END
        }
    )

    workflow.add_edge("revise_snc", "critique_snc")
    workflow.add_edge("human_approval", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


snc_graph_app = build_snc_graph()
