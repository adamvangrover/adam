# Adam Core: The Neuro-Symbolic Brain

The `core/` directory contains the cognitive architecture of the Adam system. It houses the agents, reasoning engines, and risk models that drive the "System 2" capabilities.

## 🧠 System Architecture

The Core is built on a **Neuro-Symbolic** foundation, combining the flexibility of Large Language Models (Neural) with the reliability of graph-based logic (Symbolic).

### Key Modules

*   **`agents/`**: The workforce.
    *   **Specialized Agents**: Domain experts (e.g., `RiskAnalyst`, `LegalSentinel`) that perform specific tasks.
    *   **Meta Agents**: Managers (e.g., `MetaCognitiveAgent`) that oversee and critique other agents.
*   **`engine/`**: The control center.
    *   **Neuro-Symbolic Planner**: Decomposes complex user queries into executable Task Graphs.
    *   **Meta Orchestrator**: Routes queries between Fast (Swarm) and Slow (Graph) paths.
    *   **Consensus Engine**: Resolves conflicts between agents to form a unified decision.
*   **`credit_sentinel/`**: The domain expert.
    *   A specialized module for Distressed Debt analysis, featuring 3-statement modeling and covenant extraction.
*   **`system/`**: The infrastructure.
    *   Handles asynchronous swarm coordination (`v22_async`), memory management, and context persistence.

## 🧭 The Bifurcation Protocol

As detailed in `AGENTS.md`, this codebase follows a strict bifurcation:

1.  **Product Path (`core/agents`, `core/credit_sentinel`)**:
    *   Production-grade code.
    *   Strict Pydantic typing.
    *   Auditable reasoning traces.
2.  **Lab Path (`experimental/`, `research/`)**:
    *   Rapid prototyping.
    *   Flexible schemas.

## 🔗 Integration

The Core exposes its capabilities via the **Model Context Protocol (MCP)** Server (`server/server.py`). The Web Application (`services/webapp`) and other clients interact with the Core primarily through this standardized interface.
