# Adam System 2: The Reasoning Engine

The `core/engine/` directory contains the cognitive architecture of Adam v26.0. Unlike traditional "Chain of Thought" systems that are linear, this engine uses a **Cyclical, Graph-Based** approach to reasoning.

## 🧠 Key Components

### 1. Neuro-Symbolic Planner (`neuro_symbolic_planner.py`)
**The Architect.**
*   **Role:** Decomposes a high-level user query (e.g., "Analyze Apple's credit risk") into a Directed Acyclic Graph (DAG) of executable tasks.
*   **Logic:** It uses a "Symbolic" understanding of available tools (Agents) and a "Neural" (LLM) intuition to connect them.
*   **Output:** A `TaskGraph` that defines dependencies (e.g., "Must fetch 10-K before calculating EBITDA").

### 2. Meta Orchestrator (`meta_orchestrator.py`)
**The Router.**
*   **Role:** The entry point for all queries. It decides the "Depth of Thought" required.
*   **Path A (Fast):** Routes to `core/system/v22_async` (Swarm) for simple lookups.
*   **Path B (Deep):** Routes to the `NeuroSymbolicPlanner` for complex analysis.
*   **Path C (Crisis):** Routes to `AdjudicatorEngine` for simulation scenarios.

### 3. Consensus Engine (`consensus_engine.py`)
**The Judge.**
*   **Role:** Resolves conflicts between agents.
*   **Mechanism:** If the `RiskAgent` says "Sell" but the `GrowthAgent` says "Buy", the Consensus Engine weighs their "Conviction Scores" (0-100) and produces a final Executive Decision.

### 4. Graph Definitions
*   **`cyclical_reasoning_graph.py`**: Defines the `LangGraph` state machine for the main reasoning loop (Plan -> Execute -> Critique -> Refine).
*   **`crisis_simulation_graph.py`**: A specialized graph for macro-economic stress testing.

## 🔄 The Reasoning Loop

1.  **Draft:** Agents generate an initial analysis.
2.  **Critique:** A dedicated "Critic Node" reviews the output for logical fallacies or missing citations.
3.  **Refine:** If the critique fails, the graph loops back to the execution phase with specific feedback.
4.  **Finalize:** Once the critique passes (or max iterations reached), the result is synthesized.
