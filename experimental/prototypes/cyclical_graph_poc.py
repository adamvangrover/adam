# core/system/v23_graph_engine/cyclical_graph_poc.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        draft: The current draft of the text.
        critique: The critique of the draft.
        iteration: The current iteration number.
    """
    draft: str
    critique: str
    iteration: int

def drafting_node(state: GraphState) -> dict:
    """
    Generates a draft of the text.
    """
    iteration = state.get("iteration", 0)
    if iteration == 0:
        draft = "This is the first draft."
    else:
        draft = state["draft"] + " (revised)"
    
    print(f"Drafting (Iteration {iteration}): {draft}")
    return {"draft": draft, "iteration": iteration + 1}

def critique_node(state: GraphState) -> dict:
    """
    Provides a critique of the draft.
    """
    draft = state["draft"]
    critique = "This draft is too short." if len(draft) < 30 else "This draft is good."
    print(f"Critiquing: {critique}")
    return {"critique": critique}

def should_continue(state: GraphState) -> str:
    """
    Determines whether to continue the loop.
    """
    if state["critique"] == "This draft is good.":
        return "end"
    else:
        return "continue"

# Set up the graph
graph = StateGraph(GraphState)
graph.add_node("draft", drafting_node)
graph.add_node("critique", critique_node)

# Set the entry point and the conditional edges
graph.set_entry_point("draft")
graph.add_edge("draft", "critique")
graph.add_conditional_edges(
    "critique",
    should_continue,
    {
        "continue": "draft",
        "end": END,
    },
)

# Compile and run
app = graph.compile()

if __name__ == "__main__":
    # Run the app with an empty initial state
    final_state = app.invoke({"iteration": 0})
    print("\nFinal State:")
    print(final_state)
