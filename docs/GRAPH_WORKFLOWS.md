# Graph Workflow Building Guide

## Overview
Complex workflows (e.g., "Generate a Deep Dive Credit Memo") are modeled as **Directed Acyclic Graphs (DAGs)**. This ensures deterministic execution and easier debugging compared to unstructured loops.

## Concepts
*   **Node**: A step in the process (e.g., "Fetch Data", "Summarize").
*   **Edge**: The transition between nodes (can be conditional).
*   **State**: The shared context passed between nodes.

## Building a Graph

### 1. Define State
```python
from typing import TypedDict, List

class WorkflowState(TypedDict):
    query: str
    documents: List[str]
    summary: str
```

### 2. Define Nodes
```python
def fetch_node(state: WorkflowState):
    docs = search_engine.search(state['query'])
    return {"documents": docs}

def summarize_node(state: WorkflowState):
    summary = llm.summarize(state['documents'])
    return {"summary": summary}
```

### 3. Construct Graph
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(WorkflowState)

# Add Nodes
workflow.add_node("fetch", fetch_node)
workflow.add_node("summarize", summarize_node)

# Add Edges
workflow.set_entry_point("fetch")
workflow.add_edge("fetch", "summarize")
workflow.add_edge("summarize", END)

# Compile
app = workflow.compile()
```

### 4. Execute
```python
result = app.invoke({"query": "Nvidia Earnings"})
print(result['summary'])
```
