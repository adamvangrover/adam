from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END

from core.v23_graph_engine.states import RiskAssessmentState

# --- Node Implementations ---

async def analyst_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Generates the initial financial assessment.
    """
    print("--- ANALYST NODE: Generating Draft ---")
    ticker = state["ticker"]
    data = state.get("data_context", {})
    
    # In a real implementation, this would call an LLM with the data context.
    # We simulate the output here.
    draft = f"Financial Assessment for {ticker}:\nBased on the provided data, the company shows strong liquidity."
    
    if data:
        draft += f"\nData points used: {list(data.keys())}"
        
    return {
        "analysis_draft": draft,
        "human_readable_status": "Analyst has generated the initial draft."
    }

async def reviewer_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Critiques the draft against specific credit policies.
    """
    print("--- REVIEWER NODE: Critiquing Draft ---")
    draft = state["analysis_draft"]
    critique_feedback = []
    
    # Mock critique logic
    if "Debt/EBITDA" not in draft:
        critique_feedback.append("Missing Debt/EBITDA calculation against Credit Agreement definition.")
    
    if "risk factors" not in draft.lower():
         critique_feedback.append("Analysis lacks explicit risk factor evaluation.")

    return {
        "critique_feedback": critique_feedback,
        "human_readable_status": "Reviewer has critiqued the draft."
    }

async def refinement_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Revises the draft based on critique.
    """
    print("--- REFINEMENT NODE: Refining Draft ---")
    current_draft = state["analysis_draft"]
    feedback = state["critique_feedback"]
    revision_count = state["revision_count"] + 1
    
    # Mock refinement logic
    refined_draft = current_draft + f"\n\nRevision {revision_count}:\n"
    for item in feedback:
        refined_draft += f"- Addressed: {item}\n"

    refined_draft += "Added Debt/EBITDA calculation: 3.5x.\nIncluded Risk Factors: Market volatility."
    
    return {
        "analysis_draft": refined_draft,
        "revision_count": revision_count,
        "critique_feedback": [], # Clear feedback after addressing
        "human_readable_status": f"Draft refined (Iteration {revision_count})."
    }

# --- Edge Logic ---

def should_continue(state: RiskAssessmentState) -> Literal["refinement_node", "END"]:
    """
    Decides whether to continue the refinement loop.
    """
    feedback = state["critique_feedback"]
    revision_count = state["revision_count"]
    
    if feedback and revision_count < 3:
        return "refinement_node"
    return "END"

# --- Graph Construction ---

def create_cyclical_reasoning_graph():
    workflow = StateGraph(RiskAssessmentState)
    
    # Add Nodes
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("reviewer_node", reviewer_node)
    workflow.add_node("refinement_node", refinement_node)

    # Set Entry Point
    workflow.set_entry_point("analyst_node")
    
    # Add Edges
    workflow.add_edge("analyst_node", "reviewer_node")
    
    # Conditional Edge
    workflow.add_conditional_edges(
        "reviewer_node",
        should_continue,
        {
            "refinement_node": "refinement_node",
            "END": END
        }
    )
    
    # Edge from Refinement back to Reviewer (Cyclical)
    workflow.add_edge("refinement_node", "reviewer_node")
    
    return workflow.compile()

# Export the compiled graph
cyclical_reasoning_graph = create_cyclical_reasoning_graph()
