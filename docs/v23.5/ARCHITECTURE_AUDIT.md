# Strategic Architecture Audit & Remediation Plan (Adam v23.5)

**Date:** October 26, 2024
**Status:** DRAFT
**Scope:** v23.5 "Adaptive System" Core Components

## Executive Summary
The Adam v23.5 architecture aims for a "Neuro-Symbolic" design, combining the reasoning capabilities of Large Language Models (LLMs) with the structured knowledge of Graph Databases (Neo4j/NetworkX). While the architectural vision is sound, the current implementation relies heavily on deterministic mocks, hardcoded heuristics, and "Showcase" logic.

To transition from a prototype to a production-grade "Autonomous Financial Architect," a systematic remediation is required. This document outlines the critical gaps and the step-by-step plan to address them.

---

## 1. The "Brain" Upgrade (Meta Orchestrator & Planner)

### Current State
*   **Routing:** `MetaOrchestrator` uses simple keyword matching (e.g., `if "deep dive" in query`) to route requests. This is brittle and fails on nuanced queries.
*   **Planning:** `NeuroSymbolicPlanner` uses `networkx.shortest_path` on a graph populated with hardcoded entities (e.g., defaulting to "Apple Inc."). It lacks dynamic entity discovery.
*   **Entity Extraction:** Relies on basic string checks (e.g., `if "Tesla" in request`) rather than a dedicated Named Entity Recognition (NER) pipeline.

### Remediation Plan
*   **Action 1.1: Semantic Router.** Replace `_assess_complexity` with a classifier agent (LLM-based) to determine intent (DEEP_DIVE, RISK_ALERT, MARKET_UPDATE).
*   **Action 1.2: Dynamic Entity Extraction.** Implement an extraction step in the Planner to identify Company, Sector, and Ticker from the user query using an LLM.
*   **Action 1.3: RAG-Guided Planning.** Move from `nx.shortest_path` to a retrieval-augmented generation approach where the LLM generates the navigation path based on semantic relevance.

## 2. The "Reasoning" Engine (Cyclical Graph)

### Current State
*   **Critique Logic:** The `critique_node` in `cyclical_reasoning_graph.py` uses hardcoded if/else logic based on `iteration_count`.
*   **Data Retrieval:** The `V23DataRetriever` (often mocked or hardcoded) returns fixed dictionary values for a limited set of tickers (AAPL, TSLA).
*   **State Persistence:** Uses in-memory `MemorySaver`, meaning context is lost on restart.

### Remediation Plan
*   **Action 2.1: LLM-Based Evaluator.** Replace procedural critique logic with a `SelfReflectionAgent` that prompts an LLM to review the analysis for logical fallacies and missing data.
*   **Action 2.2: Tool-Use Integration.** Connect the data retriever to live financial APIs (e.g., Yahoo Finance via `yfinance`). The agent should fallback to tool calls if data is missing.
*   **Action 2.3: State Persistence.** (Medium Priority) Implement `langgraph.checkpoint.postgres` for durable state.

## 3. Data Integrity & The "Gold Standard" Pipeline

### Current State
*   **Ingestion:** `UniversalIngestor` uses naive heuristics (e.g., text length) to assign "conviction scores."
*   **Schema:** Pydantic schemas exist (`v23_5_schema`) but are often bypassed by manual dictionary construction in fallback logic.

### Remediation Plan
*   **Action 3.1: Semantic Scoring.** Use a Cross-Encoder model to compare ingested data against ground truth for scoring.
*   **Action 3.2: Strict Pydantic Enforcement.** Ensure all graph nodes return validated Pydantic objects, raising errors on schema violations.

## 4. Infrastructure & Scalability

### Current State
*   **Dependencies:** Monolithic `requirements.txt` mixing heavy ML libs and web frameworks.
*   **Execution:** Single script execution (`scripts/run_adam.py`).

### Remediation Plan
*   **Action 4.1: Microservices Split.** (Low Priority) Separate Core Brain, Ingestion, and Simulation into distinct services.

---

## Remediation Roadmap

| Priority | Task | Target Component |
|---|---|---|
| **High** | **Dynamic Entity Extraction & Semantic Routing** | `core/engine/neuro_symbolic_planner.py` |
| **High** | **Live Data Integration (yfinance)** | `core/engine/agent_adapters.py` |
| Medium | LLM-Based Critique | `core/engine/cyclical_reasoning_graph.py` |
| Medium | Pydantic Schema Enforcement | `core/schemas/v23_5_schema.py` |
