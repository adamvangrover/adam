# core/v23_graph_engine/cyclical_reasoning_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements the core cyclical reasoning engine for Adam v23.0.
It replaces the linear v22 simulation with a stateful LangGraph workflow.
It orchestrates legacy agents via the Agent Adapters.
"""

import json
import logging
from typing import Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from core.v23_graph_engine.states import RiskAssessmentState, ResearchArtifact
from core.v23_graph_engine.agent_adapters import V23DataRetrieverAdapter, V23RiskAssessorAdapter, map_dra_to_raa

# --- Initialization ---

data_retriever = V23DataRetrieverAdapter()
risk_assessor = V23RiskAssessorAdapter()

# --- Nodes ---

def retrieve_data_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Retrieve Data
    """
    print("--- Node: Retrieve Data ---")
    ticker = state["ticker"]

    artifacts = []
    financials = data_retriever.get_financials(ticker)

    if financials:
        content = json.dumps(financials, indent=2)
        artifacts.append(ResearchArtifact(
            title=f"{ticker} Financial Data",
            content=content,
            source="V23DataRetriever",
            credibility_score=1.0
        ))
        status = f"Retrieved financials for {ticker}."
    else:
        status = f"No data found for {ticker}."

    return {
        "research_data": artifacts,
        "human_readable_status": status
    }

def generate_draft_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Generate Draft
    """
    print("--- Node: Generate Draft ---")

    financial_data_raw = {}
    for art in state["research_data"]:
        if "Financial Data" in art["title"]:
            try:
                financial_data_raw = json.loads(art["content"])
            except:
                pass

    financials, market = map_dra_to_raa(financial_data_raw)

    result = risk_assessor.assess_investment_risk(
        state["ticker"],
        financials,
        market
    )

    score = result.get("overall_risk_score", 0)
    factors = result.get("risk_factors", {})

    draft = f"RISK ASSESSMENT REPORT FOR {state['ticker']}\n"
    draft += f"Overall Risk Score: {score:.2f}\n\n"
    draft += "Risk Factors:\n"
    for k, v in factors.items():
        draft += f"- {k}: {v}\n"

    return {
        "draft_analysis": draft,
        "human_readable_status": "Drafted initial assessment.",
        "iteration_count": state["iteration_count"] + 1
    }

def critique_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Critique
    """
    print("--- Node: Critique ---")
    draft = state.get("draft_analysis", "")
    iteration = state["iteration_count"]

    critique_notes = []
    quality_score = 0.0
    needs_correction = False

    if "Error" in draft or "unavailable" in draft:
        critique_notes.append("Analysis failed due to system error.")
        quality_score = 0.1
        needs_correction = True
    elif "market_risk" not in draft.lower() and iteration < 3:
        critique_notes.append("Missing Market Risk analysis.")
        quality_score = 0.6
        needs_correction = True
    elif iteration < 2:
        critique_notes.append("Refine volatility assessment.")
        quality_score = 0.7
        needs_correction = True
    else:
        critique_notes.append("Assessment looks complete.")
        quality_score = 0.9
        needs_correction = False

    return {
        "critique_notes": critique_notes,
        "quality_score": quality_score,
        "needs_correction": needs_correction,
        "human_readable_status": "Critiqued draft."
    }

def correction_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Correction
    """
    print("--- Node: Correction ---")
    draft = state["draft_analysis"]
    notes = state["critique_notes"]

    correction_text = f"\n\n[Correction v{state['iteration_count']}]\n" + "\n".join(notes)
    new_draft = draft + correction_text

    return {
        "draft_analysis": new_draft,
        "human_readable_status": "Applied corrections.",
        "iteration_count": state["iteration_count"] + 1
    }

def human_review_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Human Review
    """
    print("--- Node: Human Review ---")
    return {
        "human_readable_status": "Awaiting Human Review."
    }

def should_continue(state: RiskAssessmentState) -> Literal["correct_analysis", "human_review", "END"]:
    quality = state["quality_score"]
    iteration = state["iteration_count"]
    needs_correction = state["needs_correction"]

    if not needs_correction or quality >= 0.85:
        return "END"

    if iteration >= 4:
        return "human_review"

    return "correct_analysis"

def build_cyclical_reasoning_graph():
    workflow = StateGraph(RiskAssessmentState)

    workflow.add_node("retrieve_data", retrieve_data_node)
    workflow.add_node("generate_draft", generate_draft_node)
    workflow.add_node("critique_analysis", critique_node)
    workflow.add_node("correct_analysis", correction_node)
    workflow.add_node("human_review", human_review_node)

    workflow.add_edge(START, "retrieve_data")
    workflow.add_edge("retrieve_data", "generate_draft")
    workflow.add_edge("generate_draft", "critique_analysis")

    workflow.add_conditional_edges(
        "critique_analysis",
        should_continue,
        {
            "correct_analysis": "correct_analysis",
            "human_review": "human_review",
            "END": END
        }
    )

    workflow.add_edge("correct_analysis", "critique_analysis")
    workflow.add_edge("human_review", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

cyclical_reasoning_app = build_cyclical_reasoning_graph()
