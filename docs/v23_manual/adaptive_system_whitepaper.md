# Adam v23.0 Adaptive System: "The Brain and the Body"

## Executive Summary
The Adam v23.0 architecture represents a paradigm shift from a purely prompt-based agent system to a **Neuro-Symbolic Hybrid**. It decouples the system into two distinct but integrated layers:

1.  **The Body (v22 Async Engine):** A high-throughput, message-driven execution layer responsible for I/O, API calls, and tool execution.
2.  **The Brain (v23 Graph Engine):** A cyclical reasoning engine responsible for planning, reflection, self-correction, and long-horizon tasks.

## Architecture

### 1. The Body: Asynchronous Message Bus
Located in `core/system/v22_async/`, the Body handles the heavy lifting.
- **Pattern:** Event-Driven (RabbitMQ/Kafka abstraction).
- **Components:** `AsyncAgentBase`, `MessageBroker`.
- **Role:** Like the autonomic nervous system, it handles reflexes and standard operations without deep thought.

### 2. The Brain: Cyclical Reasoning Graphs
Located in `core/v23_graph_engine/`, the Brain uses `LangGraph` to model thought as a graph of state transitions.

#### Key Graphs:
*   **Neuro-Symbolic Planner:** Breaks down high-level user intents into directed acyclic graphs (DAGs) of tasks.
*   **SNC Analysis Graph:** specialized for Shared National Credit logic. It uses a "Draft -> Critique -> Revise" loop to ensure regulatory compliance.
*   **Market Sentiment Graph:** A continuous monitoring loop that ingests news, updates the Knowledge Graph, and triggers alerts based on contagion risk.

### 3. The Unified Knowledge Graph (KG)
The "hippocampus" of the system.
- **Symbolic Grounding:** Uses the FIBO (Financial Industry Business Ontology) to ground LLM generations in fact.
- **Provenance:** Uses W3C PROV-O to track where every piece of data came from (e.g., "Source: Bloomberg API via Agent X").
- **Integration:** The `MarketSentimentGraph` updates the KG dynamically, linking "News Events" to "Companies" and "Sectors".

## New Capabilities in v23

### Plan-on-Graph (PoG)
Instead of linear chains-of-thought, Adam v23 generates a graph of steps. This allows for:
- **Parallelism:** Independent research tasks happen simultaneously.
- **Self-Correction:** If a node fails (e.g., "Data Missing"), the graph reroutes to a fallback node instead of crashing.

### Adversarial Red Teaming
The `RedTeamGraph` continuously attacks the system's own analysis.
- It adopts personas (e.g., "Short Seller", "Regulatory Auditor").
- It generates counter-arguments to test the robustness of financial memos.

## Developer Guide: Adding a New Graph

1.  **Define State:** Create a `TypedDict` in `core/v23_graph_engine/states.py`.
2.  **Define Nodes:** Write pure Python functions that take `State` and return a dict of updates.
3.  **Define Edges:** Use `workflow.add_edge` and `workflow.add_conditional_edges` to define the logic flow.
4.  **Register:** Add the graph to the `MetaOrchestrator` (or use it standalone).

---
*Confidential - Internal Architecture Document*
