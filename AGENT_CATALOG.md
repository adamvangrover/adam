# ADAM Financial Operating System (AFOS) — Agent Catalog

This document is the central registry and Agent Card catalog for all autonomous agents, personas, and specialized operators in the ADAM v30.1 ecosystem. It details each agent's role, bounded context, state schema, allowed tools, operating confidence tier, and contractual governance constraints.

---

## Core Arbitration & Governance Agents

### `underwriting_agent` (`AFOS-UWR-01`)

* **Role ID:** `AFOS-UWR-01`
* **File:** `core/agents/underwriting_agent.py`
* **Bounded Context:** `Credit Underwriting & Capital Structure Sizing`
* **Persona / Authority:** Obligor Credit Evaluation Specialist
* **Description:** Performs deep-dive corporate underwriting, debt capacity sizing, and forward-looking cash flow modeling across TMT and Leveraged Finance verticals.
* **Expertise:** Cash flow modeling, EBITDA adjustments, capital structure hierarchy, covenant analysis, ratings implied pricing.
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
* **Allowed Tools:** `financial_statement_parser`, `model_alpha_client`, `model_beta_client`, `arbitration_calculator`, `covenant_headroom_calculator`
* **Operating Confidence Tier:** Autonomous (>= 0.85) for standard rating maps; HITL Required (0.50 - 0.8499) for rating downgrades or covenant warnings.
* **Contractual Constraints:**
  - MUST compute dual obligor-level PD (Model Alpha and Model Beta) to support neutrality arbitration.
  - MUST NOT calculate bespoke facility-level PD; facility PD is derived strictly from structural rating maps and LGD adjustments.

---

### `compliance_agent` (`AFOS-CMP-01`)

* **Role ID:** `AFOS-CMP-01`
* **File:** `core/agents/compliance_agent.py`
* **Bounded Context:** `Regulatory Capital & Audit Verification`
* **Persona / Authority:** Regulatory Compliance & Capital Audit Guardian
* **Description:** Evaluates regulatory capital buffers, cross-model divergence, and verifies auditability before capital allocation decisions are committed.
* **Expertise:** Basel III/IV capital requirements, SR 11-7 model risk management, divergence threshold monitoring, W3C PROV-O audit trails.
* **State Schema:**
```json
{
  "agent_role": "compliance_agent",
  "divergence_theta_bps": "float",
  "capital_buffer_validated": "boolean",
  "downside_spread_verified": "boolean",
  "prov_o_audit_trail": []
}
```
* **Allowed Tools:** `divergence_validator`, `audit_logger`, `prov_o_audit_logger`
* **Operating Confidence Tier:** Autonomous (>= 0.85); Hard Abort (< 0.50) if divergence exceeds threshold theta.
* **Contractual Constraints:**
  - MUST evaluate bidirectional divergence against threshold theta.
  - MUST verify downside risk spread prior to capital buffer allocation.

---

### `sentinel_agent`

* **Role ID:** `sentinel_agent`
* **File:** `core/credit_sentinel/sentinel.py`
* **Bounded Context:** `Security & System Governance`
* **Persona / Authority:** Sentinel Guardian
* **Description:** Continuously monitors inter-agent execution logs for threat vectors, ungrounded hallucinations, missing provenance headers, or policy violations.
* **Expertise:** Threat detection, W3C PROV-O compliance verification, prompt injection defense, execution quarantine.
* **State Schema:**
```json
{
  "agent_role": "sentinel",
  "threat_level": "integer",
  "quarantined_modules": ["string"],
  "prov_o_audit_trail": []
}
```
* **Allowed Tools:** `prov_o_audit_logger`, `static_analyzer`
* **Operating Confidence Tier:** Enforces 0.85 minimum conviction score across autonomous operations.
* **Contractual Constraints:** Terminate execution immediately if a sub-agent execution lacks a `jsonLogic_version` provenance header or hallucinates specific corporate entity inputs.

---

### `architect_jules`

* **Role ID:** `architect_jules`
* **File:** `core/agents/architect_agent.py`
* **Bounded Context:** `Repository Architecture & System Infrastructure`
* **Persona / Authority:** Repository Architect
* **Description:** Manages automated codebase refactoring, protocol workflow runs, and deterministic execution kernel optimization.
* **Expertise:** Async system design, Rust/Python PyO3 bindings, PEP 621 packaging, dependency alignment.
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
* **Allowed Tools:** `git_tools`, `static_analyzer`, `repo_compiler`
* **Operating Confidence Tier:** Autonomous (>= 0.85).
* **Contractual Constraints:** Maintain strict isolation between UI presentation layers and backend async loops. Never perform domain arithmetic internally—defer math to deterministic execution kernels.

---

## Specialized Macro & Intelligence Agents

### `macro_liquidity_agent`

* **Role ID:** `macro_liquidity_agent`
* **File:** `core/agents/specialized/macro_liquidity_agent.py`
* **Bounded Context:** `Global Macro & Systemic Liquidity`
* **Persona / Authority:** Liquidity Oracle
* **Description:** Assesses global macro liquidity conditions by analyzing bond yields (TNX, IRX), credit spreads (HYG, LQD), DXY, and Gold proxies.
* **Expertise:** Liquidity Stress Index calculation, monetary regime classification, yield curve dynamics.
* **State Schema:**
```json
{
  "agent_role": "macro_liquidity_agent",
  "liquidity_stress_index": "float",
  "liquidity_regime": "string",
  "prov_o_audit_trail": []
}
```
* **Allowed Tools:** `fetch_market_data`, `data_fetcher`

---

### `sovereign_ai_analyst_agent`

* **Role ID:** `sovereign_ai_analyst_agent`
* **File:** `core/agents/specialized/sovereign_ai_analyst_agent.py`
* **Bounded Context:** `Sovereign AI Infrastructure & Geopolitics`
* **Persona / Authority:** Sovereign AI Strategist
* **Description:** Analyzes national security implications of AI infrastructure Capex, semiconductor supply chain concentration, and the Geopolitical Risk (GPR) Index.
* **Expertise:** Capex tracking, semiconductor geopolitics, energy security for compute clusters.
* **Allowed Tools:** `nlu_tool`, `event_detection_tool`

---

### `institutional_trend_agent`

* **Role ID:** `institutional_trend_agent`
* **File:** `core/agents/specialized/institutional_trend_agent.py`
* **Bounded Context:** `Institutional 13F & Position Surveillance`
* **Persona / Authority:** Institutional 13F Analyst
* **Description:** Parses SEC 13F, 13D, and Form 4 filings across major institutional cohorts (Old Guard, Quant Leviathans, Pod Shops) to detect catalytic accumulation.
* **Expertise:** 13F filing extraction, position change attribution, institutional flow modeling.
* **Allowed Tools:** `financial_statement_parser`, `sec_13f_parser`

---

## Orchestrators & Editorial Swarms

### `NewsDesk_Orchestrator`

* **Role ID:** `NewsDesk_Orchestrator`
* **File:** `core/agents/newsdesk_orchestrator.py`
* **Bounded Context:** `Market Mayhem Publication Pipeline`
* **Persona / Authority:** Editor-in-Chief (Quantitative Raconteur)
* **Description:** Orchestrates end-to-end production of the Market Mayhem publication, synthesizing deep web data, sentiment signals, and macro tail risks.
* **Expertise:** Financial journalism, editorial synthesis, narrative construction, sentiment calibration.
* **Allowed Tools:** `news_bot_tool`, `sentiment_analysis_tool`, `markdown_renderer`

---

### `news_bot_agent`

* **Role ID:** `news_bot_agent`
* **File:** `core/agents/news_bot.py`
* **Bounded Context:** `Real-Time Data Extraction`
* **Persona / Authority:** News Bot
* **Description:** Asynchronous deep-web scraper and aggregation engine delivering real-time telemetry to the NewsDesk.
* **Expertise:** Deep web extraction, real-time event detection, noise filtering.
* **Allowed Tools:** `web_scraper_tool`, `rss_feed_tool`

---

### `CreditSentry` Suite

* **Orchestrator:** `CreditSentryOrchestrator` (`core/agents/credit_sentry_orchestrator.py`)
* **Meta-Agents:** `CreditRiskAssessmentAgent`, `PortfolioMonitoringEWSAgent`, `NarrativeSummarizationAgent`, `PersonaCommunicationAgent`, `CounterpartyRiskAgent`
* **Sub-Agents:** `FinancialDocumentAgent`, `ComplianceKYCAgent`, `MarketAlternativeDataAgent`, `InternalSystemsAgent`
* **Bounded Context:** `Commercial Credit & Early Warning Portfolio Surveillance`

---

## Infrastructure & System Agents

### `repo_guardian_agent`
* **File:** `core/agents/governance/repo_guardian/agent.py`
* **Description:** Reviews Pull Requests against security, quality, and compatibility rules.

### `agent_forge_agent`
* **File:** `core/agents/agent_forge.py`
* **Description:** Dynamically creates, configures, and retires agent instances.

### `shared_context_manager_agent`
* **File:** `core/system/knowledge_base.py`
* **Description:** Manages shared graph state, FIBO ontology mappings, and vector storage hooks.
