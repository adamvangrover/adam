# Adam v23.0: Cyclical Reasoning Graph

## Introduction

Adam v23.0 introduces a powerful new architecture for building adaptive and intelligent agents: the **Cyclical Reasoning Graph**. This architecture moves beyond the linear, feed-forward message passing of v22.0 to a more flexible and dynamic model where agents can engage in iterative, reflective, and collaborative reasoning.

The core of this architecture is the ability to treat agentic workflows as stateful, cyclical graphs. This allows for:

- **Reflection & Self-Correction:** An agent's output can be routed back to itself or a "reflector" agent for iterative improvement.
- **Human-in-the-Loop (HIL) as a Node:** The graph can have nodes that explicitly pause and wait for HIL validation.
- **"Mixture-of-Agents" (MoA):** A master agent can decompose a task and spawn a sub-graph of specialist agents, wait for their aggregated reply, and then continue.

## Core Components

### `CyclicalReasoningAgent`

The `CyclicalReasoningAgent` is the primary orchestrator of cyclical reasoning workflows. It is responsible for:

- **Managing State:** Keeping track of the current state of a task, including the number of iterations remaining, the current payload, and the next agent in the graph.
- **Routing Messages:** Sending messages to the appropriate agent at each step of the workflow.
- **Terminating Execution:** Returning the final result after all iterations are complete.

### `ReflectorAgent`

The `ReflectorAgent` is a simple agent that echoes its input back to the sender. It is a useful tool for testing and debugging cyclical reasoning workflows, as it allows you to easily inspect the state of a task at each step of the process.

## Example Workflow: Iterative Improvement

Here is an example of how you can use the `CyclicalReasoningAgent` and `ReflectorAgent` to implement a simple iterative improvement workflow:

1. **Initiate the workflow:** Send an initial message to the `CyclicalReasoningAgent` with the following parameters:
   - `iterations_left`: The number of times you want to iterate.
   - `payload`: The initial data for the task.
   - `target_agent`: The name of the agent that will process the data at each step (e.g., `ReflectorAgent`).

2. **The `CyclicalReasoningAgent` processes the message:**
   - It decrements the `iterations_left` counter.
   - It sends the `payload` to the `target_agent`.

3. **The `ReflectorAgent` processes the message:**
   - It returns the `payload` to the `CyclicalReasoningAgent`.

4. **Repeat:** Steps 2 and 3 are repeated until `iterations_left` is 0.

5. **Return the result:** The `CyclicalReasoningAgent` returns the final `payload`.

This simple example demonstrates the basic principles of the Cyclical Reasoning Graph. By replacing the `ReflectorAgent` with more sophisticated agents, you can build powerful workflows for tasks such as:

- **Iterative summarization:** A document can be passed through a summarization agent multiple times to create a more concise and accurate summary.
- **Collaborative writing:** Multiple agents can work together to write a document, with each agent contributing its own expertise.
- **Automated quality assurance:** An agent can generate a piece of code, and then a "critic" agent can review it and provide feedback for improvement.

## Conclusion

The Cyclical Reasoning Graph is a major step forward for the Adam system. It provides a flexible and powerful framework for building the next generation of adaptive and intelligent agents.

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
from core.engine.cyclical_reasoning_graph import cyclical_reasoning_app
from core.engine.states import init_risk_state

initial_state = init_risk_state("AAPL", "Assess credit risk")
config = {"configurable": {"thread_id": "test_1"}}

# Run the graph
result = cyclical_reasoning_app.invoke(initial_state, config=config)

print(result["draft_analysis"])
```

