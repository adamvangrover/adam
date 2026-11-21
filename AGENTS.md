# ADAM — Agent Guide

How to build and change agents in this repository. Start with `REPO_MAP.md` (where things are) and
`docs/ARCHITECTURE.md` (how they connect). If this file disagrees with code, the code wins — fix this file.

* [1. Bifurcation: Core vs. Lab](#1-bifurcation-core-vs-lab)
* [2. Architecture: System 1 vs. System 2](#2-architecture-system-1-vs-system-2)
* [3. Operating Confidence Tiers](#3-operating-confidence-tiers)
* [4. Core Agent Role Boundaries & State Schemas](#4-core-agent-role-boundaries--state-schemas)
* [5. Standard Interface](#5-standard-interface)
* [6. Pre-Flight Checklist](#6-pre-flight-checklist)
* [7. Architectural Invariants](#7-architectural-invariants)
* [8. Extended Persona Catalog](#8-afos-agent-system-instructions--extended-persona-catalog)
* [9. v30.1-R Orchestrators](#9-v301-r-orchestrators--specialized-gate-clearance-systems)

---

## 1. Bifurcation: Core vs. Lab

Do not import Lab code into Core modules.

| Path | Locations | Rules |
|---|---|---|
| **Core** | `core/`, `services/`, `server/`, `adam_os/`, `config/`, `schemas/` | Pydantic at every I/O boundary; provenance on material decisions; defensive handling of external calls; optional heavy deps must not break imports. |
| **Lab** | `research/`, `core/experimental/` | Free iteration. Promote into `core/` by copying + testing, never by importing. |
| **Snapshots** | `archive/snapshots/<name>/` | Frozen learning snapshots, each with `SNAPSHOT.md`. Excluded from ruff, pytest and CI. Never imported. |

Presentation (`showcase/`, `frontend/`, `services/webapp/client/`) stays separate from execution code.

## 2. Architecture: System 1 vs. System 2

| | System 1 — NeuralMesh swarm | System 2 — reasoning graph |
|---|---|---|
| Path | `core/v30_architecture/python_intelligence/` | `core/agents/`, `core/engine/` |
| Base class | `agents/base_agent.py::BaseAgent` | `core/agents/agent_base.py::AgentBase` → `PydanticAgentBase` → `core/agents/templates/v30_template_agent.py::TemplateAgentV30` |
| Contract | `run()` loop; `emit(packet_type, payload)` → `NeuralPacket` | `AgentInput` → `AgentOutput` (`core/schemas/agent_schema.py`) |
| Entry | `swarm_runner.py`, `EACIOrchestrator` | `core/engine/meta_orchestrator.py::MetaOrchestrator.route_request` |
| Docs | `core/v30_architecture/README.md` | §5 below |

System 1 output is telemetry. Anything binding (ratings, sizing, capital) goes through System 2 so it
carries a `provenance_trace`. `core/system/v22_async/async_agent_base.py::AsyncAgentBase` is the older
v22 async base; new System 1 work uses the v30 `BaseAgent`.

## 3. Operating Confidence Tiers

Thresholds live in `config/agent_schema.yaml` (`TemplateAgentV30` reads them; do not hardcode).

* **Autonomous (>= 0.85):** execution permitted for low/medium-impact actions.
* **HITL Required (0.50 - 0.8499):** paused for human approval.
* **Hard Abort (< 0.50):** halt and record the failure.

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


---

## 5. Standard Interface

Single source of truth: `core/schemas/agent_schema.py` (re-exported by `core.agents.agent_base`).

```python
class AgentInput(BaseModel):
    query: str
    context: Dict[str, Any] = {}
    tools: List[str] = []

class AgentOutput(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float            # 0.0-1.0
    metadata: Dict[str, Any] = {}
    provenance_trace: List[ProvenanceHeader] = []   # empty = ungrounded; check_grounding() is False
```

The `prov_o_audit_trail` fields in the §4 state schemas are carried in `AgentOutput.provenance_trace`
(see `docs/PROVENANCE.md`).

To add a System 2 agent: subclass `TemplateAgentV30`, override `async analyze(input_data)` returning
`(answer, sources, confidence, metadata)`. The base class adds the provenance header, confidence tier,
JSONLogic guardrails and swarm logging (`AgentBase.__init_subclass__` wraps every `execute`).

Routing rules:

1. Agents never instantiate other agents. Request follow-ups via `metadata={"next_step": "...", "query": "..."}`.
2. `MetaOrchestrator` routes by ordered word-boundary patterns (`ROUTE_PATTERNS`); unresolved tickers /
   entities return `status: clarification_needed` instead of a guess.
3. Conflicting model outputs go to a consensus engine (`core/engine/consensus_engine*.py`,
   `core/engine/prov_o_consensus.py`) whose thresholds are JSONLogic/YAML.
4. The orchestrator model defaults to `gemini-4-neon` (`OrchestratorConfig`), overridden by
   `config/llm_routing.yaml` tier 1. Models in `core/config/models.py::BLOCKED_MODELS` raise
   `SR11_7_BLOCKED_MODEL_ERROR` (SR 11-7).

## 6. Pre-Flight Checklist

1. Inputs/outputs are Pydantic; no new duplicate of `AgentInput`/`AgentOutput`.
2. Thresholds and rules are in `config/*.yaml` or `assets/` JSONLogic, not in Python.
3. Hand-picked placeholder values carry `_estimated: true` (`docs/PROVENANCE.md`).
4. Tests: unit tests in `tests/unit/`; a bug fix adds a seam test in `tests/regression/`; routing/entity
   behaviour changes update `tests/evals/datasets/*.json` and pass `python tests/evals/run_accuracy.py`.
5. `.venv/bin/python -m pytest tests/ -q` and `.venv/bin/ruff check <changed files>`.
6. Record new failure patterns in `docs/AGENTS_KNOWLEDGE_BASE.md`.

## 7. Architectural Invariants

* **A:** LLMs may propose calculations; authoritative financial math runs in deterministic Python/Rust engines.
* **B:** State-changing operations pass through the governance plane (`core/governance/`, `config/governance_policy.yaml`).
* **C:** State-changing operations receive a risk classification.
* **D:** State-changing operations produce an immutable event.
* **E:** Material decisions carry provenance (`provenance_trace`, `docs/PROVENANCE.md`).
* **F:** Frozen analytical context may not silently mutate during a decision.
* **G:** Rust kernels are deterministic and independently testable (`cargo test`).
* **H:** Generated/native build artifacts are never manually edited.
* **I:** Policy lives in YAML/JSONLogic, not prompts or code.
* **J:** Tests cover success and adversarial paths.

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
- **Test Suite:** `tests/unit/test_lmt_contagion_gate_orchestrator.py`
- **Description:** Chains 5 workspace skills (`cache-search-recall` & `world-modeling`, `credit-risk-briefing` & `graph-credit-contagion`, `pd-backtesting-calibration`, `llm-as-judge-arbiter`, `librarian-archivist`) across 7 Release Gates, evaluating LMT restructuring trees (Serta uptier vs Unrestricted Sub drop-down vs Chapter 11) and counterparty spillover with SHA-256 W3C PROV-O sealing.

### B. Market Mayhem v30.1-R — Frontier AI Capex & Credit Contagion
- **Module:** `scripts/run_market_mayhem_v30_1_r.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/Market_Mayhem_v30_1_R_Frontier_AI_Capex_Prompt.md`
- **Test Suite:** `tests/unit/test_market_mayhem_v30_1_r.py`
- **Description:** Chains macro ERP signals, frontier AI lab valuation discounts, neocloud GPU SPV contagion graphs, and barbell positioning playbooks with Z-score targets and CRPS prediction bounds.

### C. IAEH & AFOS v30.1 Bitemporal Context & Policy Drift Audit
- **Module:** `scripts/run_iaeh_afos_v30_1_bitemporal_audit.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/IAEH_AFOS_v30_1_Bitemporal_Audit_Prompt.md`
- **Test Suite:** `tests/unit/test_iaeh_afos_v30_1_bitemporal_audit.py`
- **Description:** Zero-trust architecture linter inspecting cross-skill schema interfaces, bitemporal cache invalidation, and 7-Gate Release Matrix neuro-symbolic tripwires.

### D. AFOS v30.1-R2-Expanded Orchestrator
- **Module:** `scripts/run_afos_v30_1_r2_expanded_orchestrator.py`
- **Prompt Spec:** `prompt_library/AOPL-v2.0/professional_outcomes/AFOS_v30_1_R2_Expanded_Orchestrator_Prompt.md`
- **Test Suite:** `tests/unit/test_afos_v30_1_r2_expanded_orchestrator.py`
- **Description:** Evaluates exploits, defenses, and market-moving intelligence across 5 execution vectors, enforcing Lexicographic Safety Primacy (L0 > L1 > L2), VPIN order-flow toxicity filtering, zero-trust order routing, multi-oracle consensus kill-switches, and W3C PROV-O SHA-256 audit bundles.
