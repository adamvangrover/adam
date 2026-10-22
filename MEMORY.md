# Central Memory & Task Management

This document serves as the global memory buffer and task management hub across the entire Adam OS repository. It tracks active builds, pending tasks, asynchronous agent activities, swarm directives, overarching developmental goals, and core system architectures.

## 🏗️ Open Builds & Core Domains

Currently tracking active development branches, infrastructure changes, and core operational domains.
- **Adam v26.0 Stable Release:** Finalizing the hybrid cognitive engine integration.
- **Distressed Debt & Credit:** Ongoing optimization of `Credit Sentinel` for python-based 3-statement modeling, DCF valuation, SNC Rating, and Covenant Analysis.
- **Quantitative Engineering:** Enhancements to `Risk Modeling` and `Factor Analysis` for deterministic calculation of VaR, Sharpe, and Sortino ratios.
- **Logic as Data:** Expanding `jsonLogic` rulesets for decoupled risk thresholds, trading triggers, and compliance rules.
- **Rust Execution Layer:** Stability improvements for the algorithmic trading engine, matching engine, and pricing engine powered by Rust for deterministic execution.

## 🧠 System Architecture

Adam v30.1-R is architected as a **Unified Financial Operating System**, fusing System 1 neural swarms with System 2 deterministic graph planners across three distinct layers:
1. **Intelligence Layer (The "Brain"):** Houses the Neuro-Symbolic Planner (decomposes goals), the Agent Swarm (specialized domain agents), `LMTContagionOrchestratorAgent` (v30.1-R gate clearance & LMT contagion), and the Consensus Engine.
2. **Compute Layer (The "Engine"):** Houses the `LiveMockEngine` (high-fidelity market simulation), `CrisisSimulationEngine` (stress testing), and the Rust Pricing Engine.
3. **Data Layer (The "Memory"):** Handles ingestion via `Universal Ingestor` and storage via the Knowledge Graph (Neo4j) and Vector Store.

## 📋 Tasks (Prioritized Backlog)

Directly sourced from the collective Agent Knowledge Base (`docs/AGENTS_KNOWLEDGE_BASE.md`).

### P0: Security Hardening (Sentinel)
- [ ] **Audit `importlib` Usage:** Grep for dynamic imports and restrict them.
- [ ] **Fix SQL Injection:** Replace all `f"SELECT ... {var}"` with parameterized queries.
- [ ] **Secure `pickle`:** Verify no other `pickle.load` exist outside `technical_analysis.py`.
- [ ] **API Auth:** Implement Middleware for `/api/agents` and `/api/simulations`.

### P1: Architectural Refactoring (Bolt)
- [ ] **Merge Graph Classes:** Consolidate `core/engine` and `core/v23_graph_engine` versions of `UnifiedKnowledgeGraph`.
- [x] **Deduplicate Scrubbers:** Unified `GoldStandardArtifact` and `GoldStandardScrubber` in `core/data_processing/utils.py`.
- [x] **Multi-State Reasoning Grader:** Added `evals/graders/multi_state_reasoning_evaluator.py` evaluating trace fidelity, fact insertion F1, reasoning rubrics, and Expected Calibration Error.
- [x] **Archive Graph PoCs:** Moved `experimental/v23_prototypes/v23_graph_engine` to `archive/v23_prototypes/` (superseded by `core/engine/cyclical_reasoning_graph.py`).
- [ ] **Deduplicate Scrubbers:** Merge `utils.py` and `universal_ingestor.py` logic.
- [ ] **Fix Async Loggers:** Ensure all Swarm agents use append mode.

### P2: UX Improvements (Palette)
- [ ] **Accessibility Audit:** Run a linter/audit on `services/webapp/client` for `aria-` attributes.
- [ ] **Debounce Inputs:** Check all `onChange` handlers for range sliders/search inputs.

### P3: Documentation & Housekeeping
- [ ] **Update Tutorials:** Ensure `docs/tutorials/` reflect these new mandates.

## 🤖 Async Tasks, Agents, & Swarms

Tracking the autonomous activities, findings, and collaborative intelligence of the System 1 and System 2 agents.
- **Sentinel (Security):**
  - Monitoring for hardcoded secrets (e.g., `adam_api_key`) and insecure API bindings (0.0.0.0).
  - Enforcing Command Option Injection prevention by mandating the end-of-options separator (`--`) in `subprocess.run`.
  - Enforcing XXE prevention by mandating `defusedxml` over `xml.etree`.
- **Bolt (Performance):**
  - Addressing O(N) re-renders in the frontend by extracting mapped lists into `React.memo` components.
  - Mitigating list key collisions in sliding windows by assigning unique string identifiers.
  - Replacing `setInterval` with recursive `setTimeout` for async polling robustness.
- **Palette (UX):** Enforcing accessibility standards and resolving state-driven rendering issues.
- **Meta-Orchestrator Routing:** Ensuring agents communicate via the orchestrator rather than instantiating other agents directly.
- **Pheromone Trails:** Enforcing the rule that agents must log critical discoveries to `docs/AGENTS_KNOWLEDGE_BASE.md` to prevent "flash-memory amnesia."

## 🏛️ Architecture & Context Management (Governance)

Strict operational guidelines for context, tokens, orchestration, and state integration.
- **Audit Controls & Governance:** `GovernanceGatekeeper` serves as the Probabilistic-to-Deterministic Integration Layer (PDIL) to validate JSON schema and enforce confidence scoring.
- **Groundedness Tracking:** Strict W3C PROV-O compliance utilizing `ProvenanceHeader` (`wasGeneratedBy`, `generatedAtTime`, `value`) for robust audit trails.
- **Type-Safe State Management:** Standardized `AgentInput`/`AgentOutput` Pydantic schemas across all agents to maintain type safety and contextual integrity within the neuro-symbolic graph.
- **Prompt Library Organization:** Maintain and optimize Prompt Libraries via `prompt_engine.py` allowing deterministic retrieval and decoupled execution.
- **Environment Rotation:** Supporting dynamic switching between `LiveMockEngine` and Real environments for blue/green deployment and chaos engineering.

## 🧩 Machine Markers & Agent Skills (Domain Expertise)

Decomposition of monolithic models into distinct, composable capabilities for targeted domain reasoning.
- **ADAM v30.1-R Gate Clearance & LMT Contagion Orchestration:** Chained sequence of 5 skills: `cache-search-recall` & `world-modeling` (Phase 1: Bitemporal Macro Lock), `credit-risk-briefing` & `graph-credit-contagion` (Phase 2: Single-Obligor Stress, LMT Priming Tree & Syndicate GNN), `pd-backtesting-calibration` (Phase 3: Out-of-Time PD Calibration & 2LoD Floor), `llm-as-judge-arbiter` (Phase 4: Underwriter vs Challenger Debate & 7 Release Gates), and `librarian-archivist` (Phase 5: W3C PROV-O Lineage & SHA-256 Digest Seal).
- **Credit & Financial Analysis:** Implementations include `FundamentalAnalysisSkill` and `SNCRatingAssistSkill` for strict financial modeling.
- **Advanced Reasoning:** Implementations include `CounterfactualReasoningSkill` and `HybridForecastingSkill` for predictive modeling and `XAISkill` for Explainable AI output.
- **Workflow & RAG:** Implementations include `WorkflowCompositionSkill` and specific `rag_skills/` to orchestrate data retrieval dynamically.

## 📊 Evaluation & Continuous Learning

Robust testing and continuous learning to ensure agent alignment and systemic stability across simulated conditions.
- **Market Evaluation:** Evaluators like `eval_crisis_sim.py` and `eval_illiquid_market.py` ensure the system can navigate extreme financial shocks and liquidity crunches.
- **RAG & Knowledge Retrieval:** `eval_rag_pipeline.py` ensures factual grounding and precise context injection from the Knowledge Graph.
- **Unified Grading:** Leveraging `unified_eval.py` and `MultiStateReasoningEvaluator` (`evals/graders/multi_state_reasoning_evaluator.py`) to calculate trace fidelity, fact insertion F1, reasoning rubrics, and Expected Calibration Error (ECE) across multi-state reasoning pipelines.
