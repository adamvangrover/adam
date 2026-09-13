```markdown
# ADAM OS v30.1: CONSTITUTION & REPOSITORY GOVERNANCE

> **"Architecture dictates destiny. The UI observes, but the asynchronous swarm executes."**

This document serves as the constitutional baseline and system specification for the **Adam** repository (`adamvangrover/adam`). It establishes non-negotiable operational boundaries, cognitive architecture specifications, evaluation rubrics, and protocol standards across the v30.1 paradigm.

---

## 1. Core Principles & Architectural Mandates

### Article I: Environmental Bifurcation (Core vs. Lab)
The repository enforces strict separation between fiduciary execution and experimental discovery.

*   **Path A: The Core (`core/agents/`, `core/credit_sentinel/`)**
    *   **Mandate:** Deterministic execution, zero hallucination, Tier 1 G-SIB reliability.
    *   **Scope:** Underwriting, capital sizing, covenant compliance, and portfolio surveillance.
    *   **Invariants:** Strict Pydantic type safety, mandatory `try/except` wrappers around all external boundary crossings, zero unhandled exceptions, and W3C PROV-O audit trails.

*   **Path B: The Lab (`experimental/`, `research/`, `tinker_lab/`)**
    *   **Mandate:** Rapid discovery, maximal token throughput, swarm optimization.
    *   **Scope:** Novel agentic topologies, synthetic financial stress testing, and exploratory workflows.
    *   **Invariants:** Code resident in Path B **must never** be imported into Path A. Boundary violations will trigger immediate PR rejection.

### Article II: Decoupled Hybrid Cognitive Engine
Adam v30.1 rejects monolithic synchronous loops in favor of a bifurcated cognitive model:

*   **System 1 (The Neural Swarm - Autonomic Nervous System):**
    *   **Architecture:** Asynchronous, event-driven Pub/Sub (`AsyncAgentBase`).
    *   **Behavior:** Non-blocking I/O, sub-second latency, stateless execution driven by local pheromones.
    *   **Function:** Continuous surveillance of SEC EDGAR feeds, real-time market data ingestion, and detection of deal premium volatility skews.

*   **System 2 (The Neuro-Symbolic Graph - Prefrontal Cortex):**
    *   **Architecture:** Stateful Directed Acyclic Graphs (DAGs) orchestrating Temporal workflows.
    *   **Behavior:** Tool-augmented (MCP), deterministic, state-preserving, and fault-tolerant.
    *   **Function:** Multi-scenario cash flow modeling, Enterprise Value (EV) abstraction, covenant headroom calculation, and credit rating synthesis.

### Article III: Stateless Governance & Zero Hardcoded Logic
*   All credit thresholds, regulatory boundaries, and covenant tests are evaluated via stateless `jsonLogic` engines (`adam_os/contexts/governance/engine.py`).
*   Business rules, financial ratios, or underwriting criteria **must never** be embedded as conditional branching (`if/else`) inside Python execution modules.
*   All constants, discount factors, and risk weights must reside in version-controlled configuration manifests.

---

## 2. Autonomous Confidence Tiers & Decision Rubric

Every agent decision must emit an explicit confidence metric ($C \in [0.0, 1.0]$) alongside its reasoning lineage. System routing proceeds strictly according to three tiers:

*   **$\ge$ 0.85** ──► Autonomous Execution (Low/Medium Impact)
*   **0.50 to 0.8499** ──► HITL Required (Mandatory Human Operator Review)
*   **< 0.50** ──► Hard Rejection (Immediate System Abort & Log)

### System Evaluation Matrix

| Metric / Check | Target / Threshold | Action / Consequence |
| :--- | :--- | :--- |
| **Autonomous Execution** | $C \ge 0.85$ | Autonomous pipeline completion for low/medium-impact actions. W3C PROV-O audit trail generated and committed. |
| **Human-in-the-Loop (HITL)** | $0.50 \le C < 0.85$ | Pipeline execution halts safely. State is persisted in Temporal; task dispatch routes to the Nexus orchestrator for manual operator sign-off. |
| **Hard Rejection & Abort** | $C < 0.50$ | Absolute transaction abort. System issues a `CRITICAL_CONVICTION_BREACH` event, purges intermediate state, and commits diagnostics to telemetry. |
| **Bidirectional Divergence** | $\Vert{} \text{Model}_{\alpha} - \text{Model}_{\beta} \Vert{} > \theta$ | Triggers automated consensus arbitration. If unresolved, escalates to HITL review. |

---

## 3. Agent Role Boundaries & Contractual Invariants

### Core Arbitration & Underwriting Agents

*   **Underwriting Agent (`AFOS-UWR-01`)**
    *   **Authority:** Obligor credit risk assessment, cash flow forecasting, capital structure decomposition.
    *   **Mandatory Constraints:** Must calculate dual obligor-level Probability of Default (Model Alpha vs. Model Beta) to support neutrality arbitration.
    *   **Strict Prohibition:** Must not calculate bespoke facility-level PD. Facility PD is structurally derived from obligor ratings and Loss Given Default (LGD) adjustments.

*   **Compliance Agent (`AFOS-CMP-01`)**
    *   **Authority:** Regulatory capital allocation, leverage lending guidelines, audit trail validation.
    *   **Mandatory Constraints:** Must evaluate bidirectional divergence against threshold $\theta$. Must verify downside risk spreads prior to capital buffer assignment.

*   **Sentinel Agent (`AFOS-SNT-01`)**
    *   **Authority:** System security, threat modeling, provenance attestation.
    *   **Mandatory Constraints:** Must abort execution if any sub-agent payload lacks a valid `jsonLogic_version` header. Must terminate and alert upon detecting hallucinated corporate entities or synthetic market data in Path A.

*   **System Orchestrator (`Nexus`)**
    *   **Authority:** Workflow runtime, parallel task dispatch, Temporal state management.
    *   **Mandatory Constraints:** Must manage memory via Just-In-Time (JIT) retrieval against Qdrant vector memory. Must strictly prevent context stuffing (global context windows must never exceed operational baseline token budgets).

### Ecosystem Support Personas
*   **Surveillance Agent:** Tracks deal premium volatility skews; streams alerts to Temporal without blocking UI threads.
*   **Lexica Agent:** Semantic parsing and indexing of trailing 12-month SEC EDGAR filings (10-K, 10-Q, 8-K).
*   **Data Verification Agent:** Deterministic cross-validation of financial statements against primary filings.
*   **Meta-Cognitive Agent:** Continuous graph analysis scanning for disconnected nodes, reasoning loops, and logic fallacies.

---

## 4. Communication & MCP Protocols

### Article IV: Inter-Agent Interaction Rules
1.  **No Direct Instantiation:** Agents must never instantiate peer agents. All inter-agent tasks are dispatched via the `Nexus` Orchestrator using structured output metadata:
    ```python
    metadata={"next_step": "invoke_surveillance", "payload": {"target_entity": "ACME_CORP"}}
    ```
2.  **Model Context Protocol (MCP):** Agents must prioritize deterministic MCP tools (`mcp.json`) over generative reasoning for arithmetic, financial statement reconciliation, and covenant calculations.
3.  **Type-Safe Schemas:** All messages crossing agent boundaries must validate against explicit Pydantic models:

    ```python
    from pydantic import BaseModel, Field
    from typing import Dict, Any, List
    
    class AgentInput(BaseModel):
        query: str = Field(..., description="Target objective or inquiry.")
        context: Dict[str, Any] = Field(default_factory=dict, description="Shared state retrieved via JIT memory.")
        tools: List[str] = Field(default_factory=list, description="Permitted MCP tool identifiers.")
    
    class AgentOutput(BaseModel):
        answer: str = Field(..., description="Deterministic synthesis or memorandum.")
        sources: List[str] = Field(default_factory=list, description="Traceable citations (EDGAR URLs, internal IDs).")
        confidence: float = Field(..., ge=0.0, le=1.0, description="Conviction score.")
        metadata: Dict[str, Any] = Field(default_factory=dict, description="Routing and telemetry directives.")
        prov_o_audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="W3C provenance log.")
    ```

---

## 5. Data Integrity & W3C PROV-O Provenance

Every node in the **HyperDimensionalKnowledgeGraph (HDKG v30.1)** must satisfy institutional audit standards:

1.  **Cryptographic Lineage:** Every analytical assertion, financial metric, and covenant evaluation must feature a SHA-256 lineage hash referencing the raw extraction slice, filing date, and parsing agent.
2.  **Grounding Invariant:** Unreferenced assertions are treated as catastrophic hallucinations. Any node lacking a verified citation will cause pipeline failure under Sentinel inspection.
3.  **HDKG Output Conformance:** System 2 outputs must strictly conform to the `http://adam-financial.ai/schemas/v30.1/hdkg.json` schema, preserving implied facility ratings while omitting redundant bespoke PD fields.

---

## 6. Agentic Oversight Framework (AOF) & Safety Circuit Breakers

The AOF maintains autonomous operating guardrails across all environments:

*   **Recursion & Budget Breakers:** Any agent workflow exceeding 8 iterative graph cycles or consuming $>150,000$ tokens without reaching a consensus milestone is terminated automatically.
*   **Capital & Commitment Limits:** Agents are strictly forbidden from committing capital, signing advisory notices, or submitting binding regulatory reports without explicit multi-signature human approval.
*   **State Quarantine:** If an agent node produces an output below the $0.50$ confidence threshold, its intermediate artifacts are quarantined to prevent pollution of the Qdrant vector memory.

---

## 7. Contribution & CI/CD Gating Standards

To maintain Tier 1 G-SIB compliance, pull requests must satisfy these automated gates:

*   **Dependency Management:** Strict use of `uv`. Lockfiles (`uv.lock`) must remain synchronized. Pip, poetry, or conda additions are rejected.
*   **Test Isolation & Mocking:** 100% of external model calls, SEC queries, and financial APIs in test suites must be deterministically mocked. Tests must pass locally in WSL2/Linux via:
    ```bash
    uv run pytest tests/ -v --strict-markers
    ```
*   **Architecture Enforcement:** CI scripts will reject any code where:
    *   Path B (`experimental/`) modules are imported into Path A (`core/`).
    *   Business logic or credit underwriting thresholds are hardcoded in `.py` files instead of `.jsonLogic` specs.
    *   Pydantic validation is bypassed or raw dictionaries are passed across agent boundaries.

---

## 8. Constitutional Amendment Protocol

Amendments to this Constitution require:
1.  A formal Architecture Decision Record (ADR) detailing operational necessity.
2.  Dual maintainer sign-off.
3.  Schema synchronizations in `config/agent_schema.yaml` and `config/agents.yaml`.
4.  Zero regressions across the full Path A validation harness.

```
