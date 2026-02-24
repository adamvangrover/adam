from typing import TypedDict, Literal
import operator
from langgraph.graph import StateGraph, END, START

# 1. Define State
class WorkflowState(TypedDict):
    task: str
    status: str
    result: int

# 2. Define Nodes
def initialize_node(state: WorkflowState):
    print(f"[Init] Starting task: {state['task']}")
    return {"status": "initialized"}

def process_node(state: WorkflowState):
    print("[Process] Crunching numbers...")
    # Mock calculation
    return {"result": 42, "status": "processed"}

def review_node(state: WorkflowState):
    print(f"[Review] Result is {state['result']}")
    if state['result'] > 40:
        return {"status": "approved"}
    return {"status": "rejected"}

# 3. Build Graph
workflow = StateGraph(WorkflowState)

workflow.add_node("init", initialize_node)
workflow.add_node("process", process_node)
workflow.add_node("review", review_node)

workflow.add_edge(START, "init")
workflow.add_edge("init", "process")
workflow.add_edge("process", "review")
workflow.add_edge("review", END)

app = workflow.compile()

# 4. Run
if __name__ == "__main__":
    print("--- Running Custom Graph Example ---")
    inputs = {"task": "Compute The Answer", "status": "new", "result": 0}
    output = app.invoke(inputs)
    print("\n--- Final State ---")
    print(output)
