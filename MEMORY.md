# Central Memory & Task Management

This document serves as the global memory buffer and task management hub across the entire Adam OS repository. It tracks active builds, pending tasks, asynchronous agent activities, swarm directives, overarching developmental goals, and core system architectures.

## 🏗️ Open Builds & Core Domains

Currently tracking active development branches, infrastructure changes, and core operational domains.
- **AFOS v30.1 Institutional Financial OS:** Master System 1 Swarm (AsyncAgentBase / EDGAR / Unstructured) & System 2 Graph (TemplateAgentV30 / Deterministic Pricing / Covenant Execution).
- **Distressed Debt & Credit:** EVE v2.0 Institutional Underwriting Architecture (`core/engine/eve_v2_engine.py`), 5 valuation clocks, 3 independent engines, 2LoD adversarial transmission chains.
- **Quantitative Engineering:** Rust execution kernel (`rust_ext`) with GIL release, PyO3 0.23, deterministic Monte Carlo, VaR, and risk metrics.
- **Logic as Data & PDIL:** Probabilistic-to-Deterministic Integration Layer (`src/pdil/middleware.py`), JSONLogic qubit schemas, and W3C PROV-O audit trails.
- **Clean-Room Evaluation:** Multi-state reasoning evaluator (`evals/graders/multi_state_reasoning_evaluator.py`) and Clean-Room LLM-as-a-Judge grader (`evals/graders/llm_judge.py`).

## 🧠 System Architecture

AFOS v30.1 is architected as a **Dual-Process Kahneman Engine**, fusing System 1 async neural swarms with System 2 deterministic graph reasoning:
1. **Intelligence Layer (System 1 Swarm):** Asynchronous ingestion, sentiment analysis, macro pulse tracking, and parallel signal synthesis.
2. **Deterministic Compute Layer (System 2 Graph & Rust Kernel):** `rust_ext` native bindings, JSONLogic covenant engines, and multi-engine corridor convergence.
3. **Governance & Data Layer (PDIL & W3C PROV-O):** `AgentOutput` grounding verification (`check_grounding`), 3-tier conviction thresholds (>=0.85 Autonomous, 0.50-0.8499 HITL, <0.50 Hard Abort), and immutable ledgering.

## 📋 Tasks (Prioritized Backlog)

Directly sourced from the collective Agent Knowledge Base (`docs/AGENTS_KNOWLEDGE_BASE.md`).

### P0: Security Hardening & Audit (Sentinel)
- [x] **Modernize PyO3 Bindings:** Updated `rust_ext` to PyO3 0.23 with GIL releases and `rlib` crate types.
- [ ] **Audit `importlib` Usage:** Grep for dynamic imports and restrict them via `core/security/sector_validator.py` and path containment.
- [ ] **Fix SQL Injection:** Enforce parameterized queries in DuckDB/SQLite (`con.execute(query, params)`).
- [ ] **Secure `pickle`:** Maintain strict usage of `core.security.safe_unpickler.safe_load()`.

### P1: Architectural Refactoring & Evaluation (Bolt)
- [x] **Deduplicate Scrubbers:** Unified `GoldStandardArtifact` and `GoldStandardScrubber` in `core/data_processing/utils.py`.
- [x] **Multi-State Reasoning Grader:** Added `evals/graders/multi_state_reasoning_evaluator.py` evaluating trace fidelity, fact insertion F1, reasoning rubrics, and Expected Calibration Error.
- [x] **Clean-Room LLM-as-a-Judge:** Upgraded `evals/graders/llm_judge.py` with multi-criteria rubric evaluation (Correctness, Faithfulness, Reasoning, Provenance) and offline replay store.
- [x] **Archive Legacy PoCs:** Archived legacy graph engine PoCs in `archive/v23_prototypes/`.

### P2: Context Management & Persona Governance (Palette & Nexus)
- [x] **Catalog & Persona Synthesis:** Populated `AGENT_CATALOG.md` with structured agent cards and state schemas across 35+ ecosystem personas.
- [x] **Clean-Room LLM Contexts:** Synchronized `llms.txt`, `llms-full.txt`, and `AGENTS.md` to prevent prompt leakage and duplicate version generations.

## 🤖 Async Tasks, Agents, & Swarms

Tracking the autonomous activities, findings, and collaborative intelligence of the System 1 and System 2 agents.
- **Sentinel (Security):** Enforcing static security audits (`scripts/security_audit.py`), XML defused parsing, and secure URL contexts.
- **Bolt (Performance):** Releasing Python GIL in `rust_ext` during Monte Carlo and VaR calculations; optimizing PyO3 boundary conversions.
- **Palette & UI Nexus:** Maintaining clean separation between Streamlit portals, Showcase HTML dashboards, and React/Vite frontends.
- **Meta-Orchestrator Routing:** Preventing circular dependency deadlocks by enforcing routing metadata outputs through Nexus Orchestrator.

## 🏛️ Architecture & Context Management (Governance)

Strict operational guidelines for context, tokens, orchestration, and state integration.
- **Audit Controls & Governance:** Probabilistic-to-Deterministic Integration Layer (PDIL) enforces 3-tier conviction scoring and JSON logic gating.
- **Groundedness Tracking:** W3C PROV-O compliance verified via `wasGeneratedBy`, `generatedAtTime`, and `check_grounding()` validation on `AgentOutput`.
- **Type-Safe State Management:** Unified Pydantic models across `core/schemas/` and `src/schemas/core_types.py`.

## 📊 Evaluation & Continuous Learning

Robust testing and continuous learning to ensure agent alignment and systemic stability across simulated conditions.
- **Clean-Room Benchmark Harness:** `CleanRoomLLMJudge` (`evals/graders/llm_judge.py`) provides offline replay, weighted multi-criteria grading, and fallback heuristics.
- **Multi-State Reasoning Evaluator:** `MultiStateReasoningEvaluator` (`evals/graders/multi_state_reasoning_evaluator.py`) evaluates trace fidelity, fact insertion F1, reasoning rubrics, and ECE.
