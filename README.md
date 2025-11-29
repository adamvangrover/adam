# Adam v23.0 "Adaptive Hive" (Architecture Upgrade)

> **Current Status:** Transitioning from v22.0 "Monolithic Simulation" to v23.0 "Adaptive Hive".
> **Focus:** Vertical Risk Intelligence & Systems Engineering Rigor.

**[Explore the interactive demo here!](https://adamvangrover.github.io/adam/chatbot-ui/)**

## Strategic Divergence 2025: The "Adaptive Hive"

Adam v23.0 represents a paradigm shift in financial AI, moving from fragile prompt chains to a **deterministic, stateful, and self-correcting system**.

### Key Differentiators

#### 1. Cyclical Reasoning Graph (The Engine)
*   **Path A (Vertical AI):**  Instead of a linear "chain-of-thought", Adam v23.0 uses a `LangGraph` state machine.
*   **Process:** Analyst Node -> Reviewer Node (Critique) -> Refinement Node (Edit) -> Loop.
*   **Outcome:** Self-correcting analysis that doesn't hallucinate definitions.
*   **Location:** `core/v23_graph_engine/cyclical_reasoning_graph.py`

#### 2. Enterprise-Grade Data Room (MCP Integration)
*   **Connectivity:** Implements the **Model Context Protocol (MCP)** to securely connect LLMs to local data.
*   **Smart Routing:**
    *   **XBRL Path:** Precision extraction for SEC 10-Ks.
    *   **Vision Path:** VLM-based extraction for PDFs and charts.
*   **Location:** `core/vertical_risk_agent/tools/mcp_server/server.py`

#### 3. Neuro-Symbolic Planner (The Brain)
*   **Path B (Systems Engineering):** Decomposes high-level questions ("Is this company solvent?") into atomic, verifiable sub-goals.
*   **Logic:** Uses knowledge graph traversal (FIBO/PROV-O) to "discover" a reasoning path before executing it.
*   **Location:** `core/v23_graph_engine/neuro_symbolic_planner.py`

---

## Getting Started (v23.0)

### Prerequisites
*   Python 3.10+
*   `langgraph`, `mcp-python-sdk` (mocked if missing), `pydantic`

### Running the Evaluation Benchmark
Verify the agent's performance against the "Golden Set":

```bash
python evals/run_benchmarks.py
```

### Running the MCP Server
Start the financial data room server:

```bash
python core/vertical_risk_agent/tools/mcp_server/server.py
```

---

## Legacy Documentation (v21.0 - v22.0)

For historical context on the monolithic architecture, refer to:
*   [Adam v21.0 README](./docs/ui_archive_v1/README_v21.md) (Archived)
*   [v22.0 Implementation Plan](./docs/adam_v22_technical_migration_plan.md)

## Contributing

We are strictly following **Path A** (Vertical Risk) and **Path B** (Systems Engineering).
*   **Rule 1:** All agents must be typed (`Pydantic`).
*   **Rule 2:** All network calls must be `async`.
*   **Rule 3:** No linear prompt chains; use Graphs.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.
