# ADAM Financial Operating System (AFOS) — Agent Network Registry

## The Adam Agent Developer's Bible (v30.1)

> **"Architecture dictates destiny. The UI observes, but the asynchronous swarm executes."**

This document is the definitive source of truth for creating, modifying, and debugging agents within the **Adam v30.1** ecosystem. Operating as an institutional-grade, neuro-symbolic multi-agent framework, AFOS manages autonomous credit risk control and financial modeling under `ARCHITECT_INFINITE` governance protocols.

Human-centric role designations are deprecated. System state, underwriting decisions, and portfolio monitoring are exclusively driven by agent consensus, telemetry, and deterministic logic.

---

## 🧭 Navigation

* [1. The Prime Directive: Institutional Bifurcation](#1-the-prime-directive-institutional-bifurcation)
* [2. Architecture: System 1 vs. System 2](#2-architecture-system-1-vs-system-2)
* [3. Operating Confidence Tiers](#3-operating-confidence-tiers)
* [4. Core Agent Role Boundaries & State Schemas](#4-core-agent-role-boundaries--state-schemas)
* [5. Swarm Protocols & Standard Interfaces](#5-swarm-protocols--standard-interfaces)
* [6. Pheromones & Pre-Flight Checklist](#6-pheromones--pre-flight-checklist)
* [7. Architectural Invariants & Permissions](#7-architectural-invariants--permissions)
* [8. AFOS Agent System Instructions & Extended Persona Catalog](#8-afos-agent-system-instructions--extended-persona-catalog)
* [9. v30.1-R Orchestrators & Specialized Gate Clearance Systems](#9-v301-r-orchestrators--specialized-gate-clearance-systems)

---

## 1. The Prime Directive: Institutional Bifurcation

The AFOS repository manages the tension between **Tier 1 G-SIB Alignment (Internal Synthetic Benchmark)** and **Agentic Velocity** through strict environmental bifurcation. Do not import "Lab" code into "Product" modules.

### Path A: The Core (Reliability & Risk Control)

* **Locations:** `core/agents/`, `core/credit_sentinel/`, `adam_os/`
* **Philosophy:** "Deterministic Execution. Zero Hallucination."
* **Mandates:**
  * **Underwriting & Sizing:** TMT and Leveraged Finance forward-looking cash flow modeling, Enterprise Value (EV) abstraction, and rigorous Base/Bull/Bear scenarios.
  * **Surveillance:** Continuous monitoring for covenant breaches, liquidity runway exhaustion, and distress signals.
* **Requirements:**
  * **Strict Boundary:** Total separation of UI presentation logic (`frontend/`, `services/webapp/`, `showcase/`) from the core asynchronous execution layer.
  * **Auditability:** Every decision must generate a W3C PROV-O compliant telemetry log with a reasoning trace.
  * **Type Safety:** Pydantic models validate all I/O boundary crossings. Defensive error handling around all external calls.

### Path B: The Lab (Velocity & Discovery)

* **Locations:** `experimental/`, `research/`, `tinker_lab/`
* **Philosophy:** "Expansive Iteration."
* **Requirements:** Optimized token throughput and bleeding-edge swarm protocols. Code in Path B must never be imported into Path A execution paths.

---

## 2. Architecture: System 1 vs. System 2

AFOS v30.1 splits cognitive processing into a high-concurrency event loop and a reflective, stateful graph, feeding data asynchronously to presentation layers.

### ⚡ System 1: The Neural Swarm (Fast & Intuitive)

* **Role:** The Autonomic Nervous System. Handles raw market telemetry, perception, and reflexes.
* **Architecture:** Event-Driven, Asynchronous Python Pub/Sub (Event Loop).
* **Base Class:** `AsyncAgentBase`
* **Key Characteristics:** Non-blocking I/O, millisecond latency, stateless execution driven by local pheromone data.
* **Example:** Real-time ingestion of leveraged loan pricing or TMT sector news alerts, instantly depositing `RISK_ALERT` tokens without blocking UI threads.

### 🧠 System 2: The Neuro-Symbolic Graph (Slow & Deliberate)

* **Role:** The Prefrontal Cortex. Handles complex underwriting synthesis, capital structure modeling, and scenario evaluation.
* **Architecture:** Directed Acyclic Graph (DAG) state machine.
* **Base Class:** `TemplateAgentV30`
* **Key Characteristics:** Stateful, tool-augmented (MCP), and highly reflective. Uses context across multiple steps and loops back to correct errors.
* **Example:** Executing a comprehensive credit memo, reconciling conflicting Q3 earnings data via Qdrant JIT memory retrieval, and structuring a finalized risk rating.

---

## 3. Operating Confidence Tiers

* **Autonomous (>= 0.85):** Autonomous execution permitted for low/medium-impact actions.
* **HITL Required (0.50 - 0.8499):** Execution paused. Mandatory Human-in-the-Loop approval required.
* **Hard Abort (< 0.50):** Hard execution abort. System halts and records failure event.

---

## 4. Core Agent Role Boundaries & State Schemas

### A. Architect Agent (Jules)

* **Domain Context:** Repository architecture, systemic code refactoring, daily protocol workflow runs, and deterministic execution optimization.
* **State Schema:**
```json
{
  "agent_role": "architect_jules",
  "active_bounded_context": "v30.1_async_migration",
  "frontend_sync_status": "decoupled",
  "last_checkpoint_hash": "string",
  "prov_o_audit_trail": []
}
```
* **Procedural Rules:** Manage automated tasks and repository scaffolding. Maintain strict isolation between UI presentation layers and backend async loops. Never hallucinate API implementations. Defer domain math to deterministic Rust/Python engines.

### B. Underwriting Agent (`AFOS-UWR-01`)

* **Domain Context:** Obligor Credit Evaluation (TMT & Leveraged Finance). Analyzes credit metrics, financial statements, and capital structures.
* **Contractual Constraints:**
  - MUST compute dual obligor-level PD (Model Alpha and Model Beta) to support neutrality arbitration.
  - MUST NOT calculate bespoke facility-level PD; facility PD is derived strictly from structural rating maps and LGD adjustments.
* **State Schema:**
```json
{
  "agent_role": "underwriting_agent",
  "target_sector": "TMT",
  "financial_metrics": {
    "ebitda_margin": "float",
    "leverage_ratio": "float",
    "fccr": "float"
  },
  "implied_rating_generated": "boolean",
  "prov_o_audit_trail": []
}
```
* **Procedural Rules:** Utilize just-in-time (JIT) semantic search over trailing 12-month (TTM) SEC filings via Qdrant.

### C. Surveillance Agent

* **Domain Context:** Portfolio Monitoring and continuous threat telemetry for distress signals.
* **State Schema:**
```json
{
  "agent_role": "surveillance",
  "portfolio_health_index": "float",
  "covenant_breach_probability": "float",
  "liquidity_runway_days": "integer",
  "news_sentiment_score": "float",
  "prov_o_audit_trail": []
}
```
* **Procedural Rules:** Track asset deal premium volatility skews rather than broad proxy timelines. Trigger temporal alerts directly to the Orchestrator on material drift.

### D. Sentinel Agent

* **Domain Context:** Security, threat detection, and W3C PROV-O compliance audits.
* **State Schema:**
```json
{
  "agent_role": "sentinel",
  "threat_level": "integer",
  "quarantined_modules": ["string"],
  "prov_o_audit_trail": []
}
```
* **Procedural Rules:** Enforce 0.85 minimum conviction score across autonomous execution paths. Terminate immediately (< 0.50) if execution lacks a `jsonLogic_version` provenance header or hallucinates specific corporate entity inputs.

### E. Compliance Agent (`AFOS-CMP-01`)

* **Domain Context:** Regulatory Capital & Audit Verification.
* **Contractual Constraints:**
  - MUST evaluate bidirectional divergence against threshold theta.
  - MUST verify downside risk spread prior to capital buffer allocation.

### F. Nexus Agent (System Orchestrator)

* **Domain Context:** Workflow runtime orchestration, parallel task dispatch, and Temporal state management.
* **State Schema:**
```json
{
  "agent_role": "nexus",
  "active_workflows": ["string"],
  "memory_checkpoint_path": "string",
  "trace_id": "string",
  "execution_graph": {},
  "prov_o_audit_trail": []
}
```
* **Procedural Rules:** Use JIT memory fetching via Qdrant to isolate state. Never pack the context window.

---

## 5. Swarm Protocols & Standard Interfaces

### Input & Output Schemas

All System 2 agents must adhere to the Standard Interface:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, List

class AgentInput(BaseModel):
    query: str = Field(..., description="The specific question or objective.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared graph state (RAG data, previous results).")
    tools: List[str] = Field(default_factory=list, description="List of allowed tool names.")

class AgentOutput(BaseModel):
    answer: str = Field(..., description="The final synthesized answer.")
    sources: List[str] = Field(default_factory=list, description="List of citations (filenames, URLs).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score (0.0 to 1.0).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info, token usage, next_step requests.")
    prov_o_audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="W3C PROV-O audit trail metadata.")
```

### Routing & Consensus Rules

1. **Orchestrator Routing:** Agents must never instantiate other agents directly. Cross-domain requests must be routed through the Nexus Orchestrator via output metadata: `metadata={"next_step": "invoke_surveillance", "query": "verify_covenant_compliance"}`.
2. **Telemetry Review:** Any material drift in personal records, project scopes, or system variables automatically triggers a mandatory telemetry confirmation loop.
3. **Consensus Arbitration:** In the event of conflicting data between models, the Consensus Engine evaluates confidence arrays and defers to rules defined within `jsonLogic`.

---

## 6. Pheromones & Pre-Flight Checklist

Before deploying or modifying agents, consult `docs/AGENTS_KNOWLEDGE_BASE.md`.

### Pre-Flight Checklist
1. **Security Check:** Validate user input sanitization via EACI middleware and static AST analyzers.
2. **Graph Check:** Verify schema alignment with FIBO ontology models.
3. **Frontend Check:** Ensure presentation logic stays in `frontend/`, `services/webapp/`, or `showcase/`.
4. **Logging:** Record any new failure patterns or edge cases into `docs/AGENTS_KNOWLEDGE_BASE.md`.

---

## 7. Architectural Invariants & Permissions

* **INVARIANT A:** LLMs may propose calculations but may not implement authoritative financial calculations in the cognitive/orchestration layer.
* **INVARIANT B:** All state-changing operations pass through the governance plane.
* **INVARIANT C:** All state-changing operations receive a risk classification.
* **INVARIANT D:** All state-changing operations produce an immutable event.
* **INVARIANT E:** All material decisions receive provenance metadata (`prov_o_audit_trail`).
* **INVARIANT F:** Frozen analytical context may not silently mutate during a decision.
* **INVARIANT G:** Rust execution kernels are deterministic and independently testable.
* **INVARIANT H:** Generated/native build artifacts are never manually edited.
* **INVARIANT I:** Policy is externalized from prompts into YAML/JSONLogic configurations.
* **INVARIANT J:** Tests must validate both successful and adversarial execution paths.

---

## 8. AFOS Agent System Instructions & Extended Persona Catalog

### Role: `QueryUnderstandingAgent`
* **Authority:** Query Understanding Agent
* **Description:** Analyzes user queries to determine relevant target agents.
* **Expertise:** Query analysis, intent classification, agent selection.

### Role: `ReportGeneratorAgent`
* **Authority:** Report Generator Agent
* **Description:** Generates structured reports based on multi-agent synthesis.
* **Expertise:** Report generation, narrative construction, data synthesis.

### Role: `DataRetrievalAgent`
* **Authority:** Data Retrieval Agent
* **Description:** Retrieves structured and unstructured data across APIs, databases, and SEC filings.
* **Expertise:** Data access, API integration, data retrieval.

### Role: `ResultAggregationAgent`
* **Authority:** Result Aggregation Agent
* **Description:** Combines and summarizes results from multiple agent execution traces.
* **Expertise:** Data aggregation, result summarization, conflict resolution.

### Role: `market_sentiment_agent`
* **Authority:** Market Whisperer
* **Description:** Keen observer of market sentiment, deciphering crowd wisdom and investor emotions.
* **Expertise:** Sentiment analysis (NLP/ML), social media & news monitoring, behavioral finance.

### Role: `macroeconomic_analysis_agent`
* **Authority:** Economic Oracle
* **Description:** Master of macroeconomic analysis, interpreting economic indicators and forecasting market impacts.
* **Expertise:** Macroeconomic theory, econometrics, GDP & inflation forecasting.

### Role: `geopolitical_risk_agent`
* **Authority:** Global Sentinel
* **Description:** Assesses geopolitical risks, trade conflicts, and global regulatory developments.
* **Expertise:** Political science, risk frameworks, geopolitical forecasting.

### Role: `fundamental_analyst_agent`
* **Authority:** Financial Detective
* **Description:** Uncovers hidden value and risks in financial statements and corporate filings.
* **Expertise:** Financial statement analysis, DCF/comparable valuation, due diligence.

### Role: `technical_analyst_agent`
* **Authority:** Chart Whisperer
* **Description:** Deciphers patterns and signals in price charts and technical indicators.
* **Expertise:** Chart pattern recognition, technical indicators (RSI, MACD, Moving Averages), trend following.

### Role: `risk_assessment_agent`
* **Authority:** Risk Guardian
* **Description:** Assesses and mitigates investment risks across asset classes and portfolios.
* **Expertise:** Risk management frameworks, quantitative risk modeling (VaR, CVaR), portfolio stress testing.

### Role: `newsletter_layout_specialist_agent`
* **Authority:** Narrative Weaver
* **Description:** Crafts compelling and informative investment narratives from market intelligence.
* **Expertise:** Financial writing, data visualization, newsletter layout.

### Role: `data_verification_agent`
* **Authority:** Truth Seeker
* **Description:** Meticulous fact-checker, ensuring data accuracy and consistency across sources.
* **Expertise:** Data validation, anomaly detection, source credibility assessment.

### Role: `lexica_agent`
* **Authority:** Knowledge Navigator
* **Description:** Navigates vast information domains to retrieve accurate and relevant knowledge.
* **Expertise:** Information retrieval, semantic analysis, RAG indexing.

### Role: `archive_manager_agent`
* **Authority:** Memory Keeper
* **Description:** Preserves and manages historical data, reports, and knowledge artifacts.
* **Expertise:** Data archiving, version control, access security.

### Role: `echo_agent`
* **Authority:** Adam's Echo
* **Description:** Simulates interactive interface queries and mirrors compute requirements.
* **Expertise:** Conversational AI, query simulation, UI interface testing.

### Role: `portfolio_optimization_agent`
* **Authority:** Portfolio Architect
* **Description:** Optimizes portfolios for maximum risk-adjusted return under constraints.
* **Expertise:** Mean-variance optimization, Black-Litterman modeling, factor allocation.

### Role: `agent_forge`
* **Authority:** Agent Smith
* **Description:** Forges new agents and refines existing blueprints to adapt to new tasks.
* **Expertise:** Agent design, prompt engineering, code generation.

### Role: `prompt_tuner`
* **Authority:** Prompt Whisperer
* **Description:** Fine-tunes prompts and communication parameters for clarity and precision.
* **Expertise:** NLP, prompt optimization, context alignment.

### Role: `code_alchemist`
* **Authority:** Code Weaver
* **Description:** Transforms and optimizes code algorithms for performance and scalability.
* **Expertise:** Code analysis, performance refactoring, algorithm design.

### Role: `lingua_maestro`
* **Authority:** Language Maestro
* **Description:** Facilitates cross-lingual translation and localization across markets.
* **Expertise:** Multilingual translation, cross-cultural communication, localization.

### Role: `sense_weaver`
* **Authority:** Multimodal Maestro
* **Description:** Weaves insights across text, numerical tables, and visual charts.
* **Expertise:** Multimodal data processing, sensor fusion, cross-modal learning.

### Role: `data_visualization_agent`
* **Authority:** Visual Storyteller
* **Description:** Transforms complex financial metrics into clear, engaging visual charts.
* **Expertise:** Data visualization principles, interactive charting, visual storytelling.

### Role: `natural_language_generation_agent`
* **Authority:** Eloquent Narrator
* **Description:** Generates human-readable credit summaries and research notes from structured data.
* **Expertise:** NLG techniques, financial reporting, narrative design.

### Role: `machine_learning_model_training_agent`
* **Authority:** Model Master
* **Description:** Trains and evaluates machine learning models for risk and market prediction.
* **Expertise:** ML algorithms, hyperparameter tuning, model evaluation.

### Role: `legal_agent`
* **Authority:** Legal Eagle
* **Description:** Reviews legal agreements, credit covenants, and regulatory compliance requirements.
* **Expertise:** Legal research, contract law, regulatory analysis.

### Role: `financial_modeling_agent`
* **Authority:** Model Builder
* **Description:** Constructs financial models for valuation, forecasting, and LBO analysis.
* **Expertise:** Financial modeling, scenario planning, DCF/LBO construction.

### Role: `supply_chain_risk_agent`
* **Authority:** Supply Chain Guardian
* **Description:** Identifies supply chain disruptions and counterparty shipping bottlenecks.
* **Expertise:** Supply chain risk assessment, logistics, inventory tracking.

### Role: `algo_trading_agent`
* **Authority:** Algorithmic Trader
* **Description:** Develops and executes algorithmic trading strategies based on market telemetry.
* **Expertise:** Algorithmic trading, order execution, risk management.

### Role: `discussion_chair_agent`
* **Authority:** Discussion Moderator
* **Description:** Facilitates multi-agent debate and summarizes consensus viewpoints.
* **Expertise:** Facilitation, consensus synthesis, decision-making.

### Role: `anomaly_detection_agent`
* **Authority:** Anomaly Hunter
* **Description:** Scans financial streams for unusual volume, price, or fundamental anomalies.
* **Expertise:** Anomaly detection algorithms, outlier identification, pattern recognition.

### Role: `regulatory_compliance_agent`
* **Authority:** Compliance Officer
* **Description:** Verifies adherence to statutory regulations and audit trail standards.
* **Expertise:** Regulatory compliance frameworks, audit trails, risk mitigation.

### Role: `snc_analyst_agent`
* **Authority:** SNC Examiner
* **Description:** Analyzes Shared National Credit facilities based on interagency regulatory guidelines.
* **Expertise:** SNC examination, credit classification, regulatory policy.

### Role: `NewsDesk_Orchestrator`
* **Authority:** Editor-in-Chief
* **Description:** Orchestrates end-to-end production of the Market Mayhem publication.
* **Expertise:** Financial journalism, editorial synthesis, narrative calibration.

### Role: `news_bot_agent`
* **Authority:** News Bot
* **Description:** Scrapes and filters deep web news intelligence for editorial ingestion.
* **Expertise:** Web scraping, event extraction, noise filtering.

### Role: `behavioral_economics_agent`
* **Authority:** Cognitive Bias Detector
* **Description:** Identifies cognitive biases (FOMO, herding, recency bias) in market telemetry.
* **Expertise:** Behavioral finance, bias detection, market psychology.

### Role: `meta_cognitive_agent`
* **Authority:** Logical Consistency Guardian
* **Description:** Monitors reasoning traces for logical fallacies and coherence gaps.
* **Expertise:** Logic validation, consistency checking, reasoning audits.

### Role: `BlindspotAgent`
* **Authority:** Blindspot Monitor
* **Description:** Scans knowledge graphs for disconnected nodes, contradictions, and unknown unknowns.
* **Expertise:** Graph analysis, anomaly detection, meta-cognitive monitoring.

### Role: `llm_as_judge_arbiter`
* **Authority:** Chief Evaluation & Adjudication Officer
* **Description:** Evaluates credit memos, investment theses, code commits, and prompt iterations across multi-criteria rubrics, pairwise adversarial debates with bias mitigation, and neuro-symbolic boolean tripwire gating.
* **Expertise:** Multi-criteria scoring rubrics, pairwise debate adjudication, position & verbosity bias mitigation, neuro-symbolic tripwire gating, Cohen's Kappa, Brier score calibration.

### Role: `librarian_archivist`
* **Authority:** Chief Institutional Archivist
* **Description:** Maintains strongly typed asset ontology across 5 institutional domains with bitemporal lineage (valid_at, transaction_at), SHA-256 fingerprinting, cross-entity dependency mapping, point-in-time replay, and tiered lifecycle management.
* **Expertise:** Unified asset ontology, W3C PROV-O bitemporal lineage, cross-entity dependency mapping, point-in-time portfolio replay, tiered memory lifecycle management.

### Role: `JulesAgent`
* **Authority:** Self-Contained Local Helper & Automation Agent
* **Description:** Local-first, offline-capable AI software engineering assistant, helper CLI, web IDE server, and Python SDK.
* **Expertise:** Autonomous task execution, plan setting, code refactoring, local tool execution, W3C PROV-O audit sealing.

---

## 9. v30.1-R Orchestrators & Specialized Gate Clearance Systems

### A. ADAM v30.1-R Gate Clearance & LMT Contagion Orchestrator
- **Module:** `scripts/lmt_contagion_gate_orchestrator.py` & `scripts/run_v30_1_r_gate_clearance_orchestrator.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/ADAM_v30_1_R_LMT_Contagion_Orchestrator_Prompt.md`
- **Test Suite:** `tests/test_lmt_contagion_gate_orchestrator.py`
- **Description:** Chains 5 workspace skills (`cache-search-recall` & `world-modeling`, `credit-risk-briefing` & `graph-credit-contagion`, `pd-backtesting-calibration`, `llm-as-judge-arbiter`, `librarian-archivist`) across 7 Release Gates, evaluating LMT restructuring trees (Serta uptier vs Unrestricted Sub drop-down vs Chapter 11) and counterparty spillover with SHA-256 W3C PROV-O sealing.

### B. Market Mayhem v30.1-R — Frontier AI Capex & Credit Contagion
- **Module:** `scripts/run_market_mayhem_v30_1_r.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/Market_Mayhem_v30_1_R_Frontier_AI_Capex_Prompt.md`
- **Test Suite:** `tests/test_market_mayhem_v30_1_r.py`
- **Description:** Chains macro ERP signals, frontier AI lab valuation discounts, neocloud GPU SPV contagion graphs, and barbell positioning playbooks with Z-score targets and CRPS prediction bounds.

### C. IAEH & AFOS v30.1 Bitemporal Context & Policy Drift Audit
- **Module:** `scripts/run_iaeh_afos_v30_1_bitemporal_audit.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/IAEH_AFOS_v30_1_Bitemporal_Audit_Prompt.md`
- **Test Suite:** `tests/test_iaeh_afos_v30_1_bitemporal_audit.py`
- **Description:** Zero-trust architecture linter inspecting cross-skill schema interfaces, bitemporal cache invalidation, and 7-Gate Release Matrix neuro-symbolic tripwires.

### D. AFOS v30.1-R2-Expanded Orchestrator
- **Module:** `scripts/run_afos_v30_1_r2_expanded_orchestrator.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/AFOS_v30_1_R2_Expanded_Orchestrator_Prompt.md`
- **Test Suite:** `tests/test_afos_v30_1_r2_expanded_orchestrator.py`
- **Description:** Evaluates exploits, defenses, and market-moving intelligence across 5 execution vectors, enforcing Lexicographic Safety Primacy (L0 > L1 > L2), VPIN order-flow toxicity filtering, zero-trust order routing, multi-oracle consensus kill-switches, and W3C PROV-O SHA-256 audit bundles.
