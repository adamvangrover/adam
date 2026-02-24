# Tutorial: Building a LangGraph Workflow

Adam v26.0 uses **LangGraph** to model complex, multi-step reasoning processes (System 2). This tutorial shows how to build a simple "Research & Critique" graph.

## Concepts
*   **State:** A typed dictionary that holds all data for the workflow run.
*   **Nodes:** Functions that modify the state.
*   **Edges:** Rules for moving between nodes.

## Step 1: Define State
Create a Pydantic model or TypedDict for your graph state.

```python
from typing import TypedDict, List

class ResearchState(TypedDict):
    topic: str
    draft: str
    critique: str
    revision_count: int
```

## Step 2: Define Nodes

```python
def drafter_node(state: ResearchState):
    print(f"Drafting content for {state['topic']}...")
    return {"draft": "Initial draft content..."}

def critic_node(state: ResearchState):
    print("Critiquing draft...")
    if "bad" in state["draft"]:
        return {"critique": "Too negative."}
    return {"critique": "Looks good."}
```

## Step 3: Build the Graph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("drafter", drafter_node)
workflow.add_node("critic", critic_node)

# Add Edges
workflow.set_entry_point("drafter")
workflow.add_edge("drafter", "critic")
workflow.add_edge("critic", END)

# Compile
app = workflow.compile()
```

## Step 4: Execution

```python
inputs = {"topic": "AI Safety", "revision_count": 0}
result = app.invoke(inputs)
print(result)
```

## Advanced: Conditional Edges
To add a loop (Draft -> Critic -> Retry -> Draft):

```python
def should_continue(state):
    if state["critique"] == "Looks good.":
        return END
    return "drafter"

workflow.add_conditional_edges("critic", should_continue)
```

See `core/engine/deep_dive_graph.py` for a production example.
