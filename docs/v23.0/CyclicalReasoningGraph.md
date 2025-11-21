# Cyclical Reasoning Graph (v23.0)

## Overview
The **Cyclical Reasoning Graph** is the core execution engine of the Adam v23.0 "Adaptive" architecture. Unlike the linear, prompt-driven simulations of v22.0, this engine uses `LangGraph` to create a stateful, iterative workflow that supports:
- **Self-Correction:** The system generates a draft, critiques it, and automatically attempts to fix errors before showing the user.
- **Human-in-the-Loop (HIL):** If the system fails to self-correct after a defined number of attempts, it pauses and escalates to a human reviewer.
- **State Persistence:** The entire reasoning process is stored in a state object, allowing for auditability and "time-travel" debugging.

## Architecture

### 1. State Object (`RiskAssessmentState`)
The graph's memory is a `TypedDict` that tracks:
- `ticker`: The subject of analysis.
- `draft_analysis`: The current version of the report.
- `critique_notes`: Feedback from the Reflector Agent.
- `iteration_count`: How many times the system has tried to fix the report.
- `quality_score`: A numerical score of the current draft's validity.

### 2. Nodes
- **Retrieve Data:** Fetches raw financial data (simulated via `DataRetrievalAgent`).
- **Generate Draft:** Creates the initial analysis (simulated via `RiskAssessmentAgent`).
- **Critique:** Reviews the draft for logical errors, missing data, or hallucinations (simulated via `ReflectorAgent`).
- **Correction:** Modifies the draft based on the critique notes.
- **Human Review:** A breakpoint node that halts execution if quality standards are not met.

### 3. Conditional Logic (The "Inner Loop")
The graph uses conditional edges to determine the flow:
1.  **Generation** -> **Critique**
2.  **Critique** -> **Decision**:
    *   If `Quality > Threshold`: -> **END** (Success)
    *   If `Quality < Threshold` AND `Iterations < Max`: -> **Correction** -> **Critique** (Loop)
    *   If `Quality < Threshold` AND `Iterations >= Max`: -> **Human Review** (Failure/Escalation)

## Usage

```python
from core.v23_graph_engine.cyclical_reasoning_graph import cyclical_reasoning_app
from core.v23_graph_engine.states import init_risk_state

initial_state = init_risk_state("AAPL", "Assess credit risk")
config = {"configurable": {"thread_id": "test_1"}}

# Run the graph
result = cyclical_reasoning_app.invoke(initial_state, config=config)

print(result["draft_analysis"])
```
