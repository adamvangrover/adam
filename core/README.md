# Adam Core Engine

The `core/` directory is the heart of the Adam system, containing the "System 2" reasoning engine, agents, and data processing pipelines.

## 🧠 System 2 Reasoning Engine

Adam v26.0 moves beyond simple "Chain-of-Thought" prompting to a **Cyclical Reasoning Architecture** (System 2). This is implemented using `LangGraph` in `core/engine/`.

### Key Concepts

*   **Neuro-Symbolic Planner:** (`core/engine/neuro_symbolic_planner.py`) Decomposes high-level goals into a Directed Acyclic Graph (DAG) of tasks.
*   **Meta Orchestrator:** (`core/engine/meta_orchestrator.py`) The central router that directs queries to the appropriate execution engine (System 2 Graph vs. Async Swarm).
*   **Consensus Engine:** (`core/engine/consensus_engine.py`) Aggregates signals from multiple agents to form a final "Executive Decision".

## 📂 Directory Structure

| Directory | Description |
| :--- | :--- |
| `agents/` | Contains the specialized agents (e.g., Risk Analyst, Fundamental Analyst). |
| `analysis/` | Tools for deep financial analysis (Fundamental, Technical, Sentinel). |
| `data_processing/` | The "Universal Ingestor" pipeline for scrubbing and normalizing data. |
| `engine/` | The core reasoning logic, including the Planner, Orchestrator, and Graph definitions. |
| `simulations/` | Environments for running scenarios (e.g., Crisis Simulation). |
| `system/` | System-level utilities, logging, and the Async Swarm infrastructure. |

## 🚀 Getting Started with Core

To interact with the core engine directly (for development or debugging), you can use the `scripts/run_adam.py` entry point or the provided Jupyter notebooks.

### Running a Deep Dive

```bash
# From the repository root
python scripts/run_adam.py --mode deep_dive --ticker AAPL
```

## ⚠️ Agents & Developers

Please refer to `AGENTS.md` in this directory for specific coding standards and "Rules of Engagement" when modifying the core engine.
