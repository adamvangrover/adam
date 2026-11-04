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

### P0: Security & Zero-Trust Enforcement (Sentinel)
- [x] **AFOS V3 Zero-Trust Ingestion Gateway:** Implemented mTLS (TLSv1.3), 15-min ECDSA STS tokens, W3C PROV-O SHA-256 bit-mismatch quarantine, and 3-feed 0.5sigma consensus with 75/25 Barbell LOCKDOWN circuit breaker.
- [x] **Audit `importlib` & Secure `pickle`:** Replaced `pickle.load` / `pickle.loads` with `core.security.safe_unpickler` and restricted dynamic imports.
- [x] **Fix SQL Injection:** Replaced `f"SELECT ... {var}"` with parameterized queries, tokenizer SELECT validation, and `sqlite3.set_authorizer`.
- [x] **SSRF Redirect & Jinja2 XSS Prevention:** Disabled automatic redirects in `SecurityGovernanceGatekeeper` and enabled `autoescape=True` in `build_dashboard.py`.
- [x] **API Auth & Adaptive JWT Fallback:** Enforced `X-API-Key` and Adaptive JWT fallback across command endpoints.
- [ ] **Continuous `importlib` & `pickle` CI Gate:** Enforce automated CI gate blocking any new `pickle.load` or unwhitelisted `importlib.import_module` across `core/` and `src/`.

### P1: Kernel, Architecture & Performance (Bolt)
- [x] **ADAM-IAK-CORE Primitives (`src/iak/`):** Remediated `DtypeSafeMaskedEntmaxSoftmax`, `UncertaintyGatedCRFSlotDecoder`, and `DeterministicPolicyEpistemicGate`.
- [x] **Pre-Compiled JSONSchema & Vectorized DataFrames:** Eliminated `df.iterrows()` in `WhaleScanner`/`DataFetcher` and pre-compiled `jsonschema` validators in `eval_001a_verifier.py`.
- [x] **Deduplicate Scrubbers & Archive Graph PoCs:** Unified `GoldStandardArtifact` and `GoldStandardScrubber` in `core/data_processing/utils.py`; moved `experimental/v23_prototypes/v23_graph_engine` to `archive/v23_prototypes/`.
- [x] **Multi-State Reasoning Grader:** Added `evals/graders/multi_state_reasoning_evaluator.py` evaluating trace fidelity, fact insertion F1, reasoning rubrics, and ECE.
- [x] **Fix Async & Append-Only Loggers:** Ensured all Swarm and Jules agents use append mode (`open(..., 'a')`).
- [ ] **Consolidate Legacy Graph Imports:** Complete migration of remaining `core/v23_graph_engine` references into `core/engine/unified_knowledge_graph.py`.
- [ ] **Complete Scrubber Consolidation:** Merge any remaining duplicate text-cleaning logic in `universal_ingestor.py` into `core/data_processing/utils.py`.

### P2: UX & Accessibility (Palette)
- [x] **Debounce Inputs & Split Fuse.js Memoization:** Debounced simulation range sliders and separated Fuse.js index instantiation from search execution.
- [x] **Daily Hub Index Synchronization:** Updated `showcase/adam_daily_hub_index.html` and provenance checks in `ops/checks/check_provenance.py`.
- [ ] **ARIA Live & Roving TabIndex Audit:** Run a full accessibility audit across `services/webapp/client` to verify `role="log"`, `aria-live="polite"`, keyboard roving `tabIndex`, and stable UUID keys across all streaming widgets.

### P3: Documentation & Governance Housekeeping
- [x] **Update Tutorials & PDIL Mapping:** Added `docs/architecture/PDIL_MAPPING.md`, `docs/jules_tools_reference.md`, and `docs/how-to/using_jules_cli_ide_sdk.md`.

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
