```markdown
# ADAM OS v30.1: CONSTITUTION & REPOSITORY GOVERNANCE

> **"Architecture dictates destiny. The UI observes, but the asynchronous swarm executes."**

This document serves as the constitutional baseline and system specification for the **Adam** repository (`adamvangrover/adam`). It governs all agent behavior, architectural boundaries, and evaluation criteria across the v30.1 paradigm. 

---

## 1. Core Principles & Architectural Mandates

### Article I: Environmental Bifurcation (Core vs. Lab)
The repository enforces strict isolation between production stability and experimental velocity.
*   **Path A: The Core (`core/agents/`, `core/credit_sentinel/`)**
    *   *Philosophy:* Deterministic execution, zero hallucination, and Tier 1 G-SIB reliability standards.
    *   *Rules:* Strict Pydantic type safety, mandatory `try/except` wrappers around external calls, and W3C PROV-O compliant telemetry logging for every decision.
*   **Path B: The Lab (`experimental/`, `research/`, `tinker_lab/`)**
    *   *Philosophy:* Expansive iteration, optimized token throughput, and bleeding-edge swarm mechanics.
    *   *Rules:* Code here is strictly siloed and **must never** be imported into Path A.

### Article II: The Hybrid Cognitive Engine
Adam v30.1 abandons synchronous blocking loops in favor of a decoupled cognitive model:
*   **System 1 (The Neural Swarm):** Asynchronous Python Pub/Sub (`AsyncAgentBase`) operating on local pheromones. Handles low-latency market telemetry, perception, and deal premium volatility skews without blocking UI threads.
*   **System 2 (The Neuro-Symbolic Graph):** Stateful Directed Acyclic Graphs (DAGs) powered by Temporal workflows and Qdrant JIT vector memory. Handles complex credit underwriting, debt covenant modeling, and synthesized risk ratings.

### Article III: Stateless Governance & Zero Magic Numbers
*   All credit risk thresholds, covenant evaluation logic, and compliance triggers must be evaluated via stateless `jsonLogic` engines. 
*   Business rules must never be hardcoded as procedural branching within Python modules.

---

## 2. Autonomous Confidence Tiers & Evaluation Rubric

All agent operations are governed by three rigorous conviction thresholds:


```

[Confidence Score >= 0.85] ──► Autonomous Execution (Low/Medium Impact)
[0.50 <= Score < 0.85]     ──► HITL Required (Mandatory Human Operator Review)
[Score < 0.50]             ──► Hard Rejection (Immediate System Abort & Log)

```

### Evaluation Matrix

| Metric / Check | Target / Threshold | Action / Consequence |
| :--- | :--- | :--- |
| **Autonomous Execution** | $\ge 0.85$ Confidence | Permitted for low/medium-impact actions with valid W3C PROV-O audit trails. |
| **Human-in-the-Loop (HITL)** | $0.50$ to $0.8499$ | Execution pauses safely; routed to the Nexus orchestrator for manual sign-off. |
| **Hard Rejection & Abort** | $< 0.50$ Conviction | Immediate abort of the agent node, failure logging, and telemetry preservation. |

---

## 3. Agent Role Boundaries & Contractual Constraints

### Core Specialists
*   **Underwriting Agent (`AFOS-UWR-01`)**
    *   *Domain:* TMT and Leveraged Finance credit analysis, cash flow modeling, Enterprise Value (EV) abstraction.
    *   *Constraint:* Must compute dual obligor-level Probability of Default (Model Alpha and Model Beta) for neutrality arbitration. **Must not** calculate bespoke facility-level PDs directly (facility PD is structurally derived from rating maps).
*   **Compliance Agent (`AFOS-CMP-01`)**
    *   *Domain:* Regulatory capital and audit verification.
    *   *Constraint:* Must evaluate bidirectional divergence against threshold $\theta$ and verify downside risk spreads before capital allocation.
*   **System Orchestrator (`Nexus`)**
    *   *Domain:* Workflow runtime, parallel task dispatch, and Temporal state management.
    *   *Constraint:* Must use just-in-time (JIT) memory fetching via Qdrant to prevent context window bloat. Direct agent-to-agent instantiation is strictly prohibited; cross-domain requests must use metadata routing (`next_step`).

---

## 4. Communication & MCP Protocols

1.  **I/O Schema Enforcement:** All agent inputs and outputs must validate against strict Pydantic definitions (`AgentInput`, `AgentOutput`).
2.  **Model Context Protocol (MCP):** Agents must prioritize registered Tools defined in `mcp.json` over internal arithmetic or unverified calculations.
3.  **No Direct Instantiation:** Agents must never instantiate peer agents directly. Requests for cross-domain processing must be delegated back through the `Nexus` orchestrator via output metadata instructions.
```markdown
metadata={"next_step": "invoke_surveillance", "query": "Monitor covenant headroom"}

```

```

```
