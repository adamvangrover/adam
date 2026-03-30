# ROADMAP: CONTINUOUS REPOSITORY MODERNIZATION

## Overview
The "Merge & Purge" methodology establishes a structured approach to technical debt remediation. This roadmap outlines the strategic phases for consolidating legacy architectural paradigms within the Cognitive Financial Operating System into unified execution kernels.

## Phase 1: Strangulation of Legacy v23 Components (COMPLETED)
*   **Target:** `core/v23_graph_engine`
*   **Action:** Merged logic to `core/engine/` and purged the legacy directory.
*   **Impact:** Centralized the "System 2" reasoning graph, unifying the `meta_orchestrator`, `unified_knowledge_graph`, and `simulation_engine` under a single namespace. Verified via complete test execution.

## Phase 2: System 1 Perception Consolidation (IN PROGRESS/UPCOMING)
*   **Target:** `core/system1`, `core/data_processing`, `core/data_sources`
*   **Objective:** Unify data ingestion, chunking, and primary perception layers into a cohesive "Observation Lakehouse" pipeline.
*   **Action:**
    1. Identify redundant Universal Ingestors (e.g., `universal_ingestor.py` vs `universal_ingestor_v2.py`).
    2. Consolidate ETL pathways into the designated high-performance Python/Rust ingestor pipeline.
    3. Purge deprecated logic and enforce a single `DataLoader` pattern (as seen in `core/utils/data_utils.py`).
*   **Oversight Level:** HITL (Human-In-The-Loop) required prior to the purge of primary API ingestion modules.

## Phase 3: Risk Model & Valuation Synchronization
*   **Target:** `core/risk_engine`, `core/valuation_utils.py`, `core/pricing_engine`
*   **Objective:** Eliminate disparate Python-based quantitative pricing models and formally shift execution to the high-frequency Rust bindings (`core/rust_pricing/`).
*   **Action:**
    1. Identify legacy Avellaneda-Stoikov or Monte Carlo implementations in standard Python logic.
    2. Reroute AI Swarm queries (via MCP tools) to interface exclusively with the high-performance Rust execution engine.
    3. Deprecate and purge legacy probabilistic pricing generators.
*   **Oversight Level:** HITL (Human-In-The-Loop) - Crucial for ensuring that live algorithmic order flow remains uncompromised during the transition.

## Phase 4: Swarm Protocol Standardization
*   **Target:** `core/agents/`, `core/swarms/`
*   **Objective:** Enforce the Hybrid Neurosymbolic Agent State Protocol (HNASP) across all distributed agents.
*   **Action:**
    1. Audit all individual specialized agents for non-compliant states (e.g., missing Pydantic V2 schemas).
    2. Standardize all agent interactions onto the centralized `MessageBroker` and asynchronous event loop.
    3. Purge "Frankenstein" agents that bypass the `meta_orchestrator` execution graph.
*   **Oversight Level:** HOTL (Human-On-The-Loop) - Automated linting and test validation should govern the majority of these refactors.

## Conclusion
The aggressive execution of this roadmap guarantees that the repository remains agile, deterministic, and highly scalable. By replacing the N-tier monolith with a polyglot microservices architecture orchestrated via Kubernetes, the system ensures optimal latency profiles for real-time market data ingestion and continuous agentic reasoning.
