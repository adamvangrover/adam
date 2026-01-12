import logging
from typing import TypedDict, Annotated, List, Dict, Any, Union
import operator

# LangGraph Imports
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver # We use MemorySaver here, but interface allows Postgres
except ImportError:
    StateGraph = None
    END = "END"
    START = "START"
    MemorySaver = None

from core.v24_architecture.reasoning.self_reflection import SelfReflectionAgent

logger = logging.getLogger(__name__)

# --- State Definition ---

class ReasoningState(TypedDict):
    """
    The persistent state for the reasoning loop.
    """
    request: str
    thread_id: str
    current_draft: str
    critique_feedback: List[str]
    iteration_count: int
    quality_score: float
    is_complete: bool
    schema_version: str # For future migration safety

# --- Nodes ---

async def draft_node(state: ReasoningState):
    """
    Agent that drafts the analysis.
    """
    logger.info(f"Drafting Node (Iteration {state['iteration_count']})")

    # Mock drafting logic
    current = state.get("current_draft", "")
    feedback = state.get("critique_feedback", [])

    if not current:
        draft = f"Analysis for {state['request']}: \n- Market looks volatile."
    else:
        draft = current + f"\n- Addressing feedback: {feedback[-1] if feedback else ''}"

    return {
        "current_draft": draft,
        "iteration_count": state["iteration_count"] + 1
    }

async def reflection_node(state: ReasoningState):
    """
    Agent that critiques the draft.
    """
    logger.info("Reflection Node")
    agent = SelfReflectionAgent()
    result = await agent.critique(state["current_draft"], {})

    return {
        "quality_score": result["score"],
        "critique_feedback": result["feedback"],
        "is_complete": result["status"] == "PASS"
    }

# --- Conditional ---

def should_continue(state: ReasoningState):
    if state["is_complete"]:
        return "end"
    if state["iteration_count"] > 3: # Max loops
        return "end"
    return "continue"

# --- Graph Builder ---

def build_robust_graph():
    if not StateGraph:
        logger.error("LangGraph not installed.")
        return None

    workflow = StateGraph(ReasoningState)

    workflow.add_node("drafter", draft_node)
    workflow.add_node("critic", reflection_node)

    workflow.add_edge(START, "drafter")
    workflow.add_edge("drafter", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "continue": "drafter",
            "end": END
        }
    )

    # In production, we would pass a PostgresSaver here
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
