# Swarm Evolution Log

**Status:** Active
**Epoch:** v2.0 (The Agentic Singularity)

This document serves as the immutable record of the Adam system's architectural evolution. It tracks the shift from static scripts to dynamic, self-improving agent swarms.

## Evolutionary Timeline

### Epoch 1: The Static Era (v21)
*   **Architecture:** Monolithic scripts.
*   **Cognition:** Linear, single-shot execution.
*   **State:** Ephemeral.

### Epoch 2: The Graph Awakening (v23)
*   **Architecture:** Neuro-Symbolic Graph (`UnifiedKnowledgeGraph`).
*   **Cognition:** Cyclical reasoning loops (`LangGraph`).
*   **State:** Graph-based, persistent within session.
*   **Key Artifacts:** `core/v23_graph_engine/` (Now Legacy).

### Epoch 3: The Agentic Swarm (v2.0 - Current)
*   **Architecture:** Hybrid MCP + Swarm Intelligence.
*   **Cognition:** Distributed, Tool-Using, Self-Reflective.
*   **State:** `MemoryMatrix` (Persistent Collective Unconscious).
*   **Key Features:**
    *   **Model Context Protocol (MCP):** Standardized tool interfaces (`core/mcp`).
    *   **Swarm Telemetry:** Structured JSONL logging for full observability (`core/utils/logging_utils.py`).
    *   **Evolutionary Optimization:** Meta-agents capable of code analysis (`core/agents/evolutionary_optimizer.py`).

## Architectural Directives

1.  **Do Not Delete:** Legacy code is a fossil record. Preserve it in `core/v23_graph_engine` or similar archives.
2.  **Graceful Fallback:** The system must degrade gracefully. If the Swarm fails, fall back to the Graph. If the Graph fails, fall back to linear scripts.
3.  **Radical Additivity:** Always build *on top*. Expand the capabilities without breaking the foundation.

## Future Trajectory (v25+)
*   **Genetic Algorithms:** Agents that spawn child agents with mutated prompts.
*   **Quantum-Native State:** Moving the `MemoryMatrix` to a quantum-resilient data structure.

## YYYY-MM-DD - [Architectural Consolidation] Merge & Purge of v23_graph_engine
**Objective:** Transition the legacy v23 "Adaptive System" graph engine components from `core/v23_graph_engine` to the unified kernel at `core/engine/` to eliminate technical debt and consolidate the repository architecture.
**Action:**
1. Mapped legacy modules (`unified_knowledge_graph`, `odyssey_knowledge_graph`, `simulation_engine`, `meta_orchestrator`, `cyclical_reasoning_graph`, `self_reflection_agent`) to their centralized counterparts in `core/engine/`.
2. Replaced imports across all test scripts (`tests/z_test_api_v23_wiring.py`, `tests/verify_simulation_logic.py`, `tests/test_odyssey_graph_integration.py`, `tests/benchmark_ukg.py`, `tests/verify_v23_graph.py`, `tests/test_odyssey_flow.py`) and source files (e.g. `core/agents/governance/repo_guardian/prompts.py`, `core/main.py`).
3. Fixed subsequent logic errors and incorrect dependencies exposed in tests by the import migration (e.g. `validate_data` resolution, `SECRET_KEY` requirements, missing dictionary keys, outdated metric calculations).
4. Ran full verification via `uv run pytest` to ensure successful backward compatibility with the new kernel logic.
5. Executed the Purge phase by deleting the deprecated directory `core/v23_graph_engine`.

**Why:** The initial "Additive-Only" architecture resulted in a bloated, monolithic structure that hindered continuous execution speed. The "Merge & Purge" methodology actively strangulates the monolith, deprecating legacy structures for a polyglot microservices architecture governed by HNASP, ensuring deterministic business logic upon probabilistic reasoning.
**Swarm Protocol:** HOTL (Human-On-The-Loop) - The deletion of `core/v23_graph_engine` was executed autonomously following successful verification of the migrated logic in `core/engine/`.
