# core/engine/reflector_graph.py

import logging
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from core.engine.states import ReflectorState

logger = logging.getLogger(__name__)

# --- Mock Logic for Nodes ---


def mock_analyze_content(content: str) -> Dict[str, Any]:
    notes = []
    score = 1.0

    if len(content) < 50:
        notes.append("Content is too short.")
        score -= 0.5

    if "error" in content.lower():
        notes.append("Content contains error messages.")
        score -= 0.5

    return notes, score


def mock_refine_content(content: str, notes: list) -> str:
    return content + "\n\n[Refined based on feedback: " + "; ".join(notes) + "]"

# --- Nodes ---


def analyze_node(state: ReflectorState) -> Dict[str, Any]:
    logger.info("--- Node: Analyze Content ---")
    notes, score = mock_analyze_content(state.get("refined_content") or state["input_content"])

    is_valid = score > 0.8

    return {
        "critique_notes": notes,
        "score": score,
        "is_valid": is_valid,
        "human_readable_status": f"Analyzed content. Score: {score}"
    }


def refine_node(state: ReflectorState) -> Dict[str, Any]:
    logger.info("--- Node: Refine Content ---")
    current_content = state.get("refined_content") or state["input_content"]
    new_content = mock_refine_content(current_content, state["critique_notes"])

    return {
        "refined_content": new_content,
        "iteration_count": state["iteration_count"] + 1,
        "human_readable_status": "Refined content based on critique."
    }

# --- Conditional ---


def should_continue_reflection(state: ReflectorState) -> Literal["refine", "finalize"]:
    if not state["is_valid"] and state["iteration_count"] < 3:
        return "refine"
    return "finalize"

# --- Graph ---


def build_reflector_graph():
    workflow = StateGraph(ReflectorState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("refine", refine_node)

    workflow.add_edge(START, "analyze")

    workflow.add_conditional_edges(
        "analyze",
        should_continue_reflection,
        {
            "refine": "refine",
            "finalize": END
        }
    )

    workflow.add_edge("refine", "analyze")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


reflector_app = build_reflector_graph()
