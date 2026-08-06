# Adam Financial Operating System (AFOS) Product Specification

## 1. Vision
The Adam Financial Operating System (AFOS) represents an architectural inflection point. We are transitioning from an "Autonomous Multi-Agent Risk Framework" (an AI-centric approach) to a "Financial Decision Operating System" (a decision-centric institutional infrastructure).

Under AFOS, Large Language Models (LLMs) and agents are demoted from architectural primitives to mere implementation details—replaceable execution engines constrained by immutable policies and ontologies.

## 2. The Architectural Inversion
Legacy AI wrappers implicitly assume:
`Workflow → Decision`

AFOS explicitly enforces:
`Decision → Evidence → Policy → Execution → Workflow`

In this formulation, workflows become orchestration around immutable decisions rather than the source of truth.

## 3. Financial Applications
AFOS serves as the foundational infrastructure for multiple institutional applications, all of which inherit the same kernels and ontology:
* **Credit Risk:** Underwriting, deep-dive credit memos, rating assignments.
* **Portfolio Management:** Surveillance, aggregation, concentration limits.
* **Loan Monitoring:** Automated covenant compliance and watchlist triggers.
* **LevFin & Syndications:** Deal structuring and capital allocation.
* **Treasury & Wealth Management:** Alpha generation and scenario modeling.
* **Regulatory Reporting:** Automated, PROV-O compliant audit trails.

## 4. The Seven Foundational Kernels
The operating system is partitioned into seven decoupled kernels:
1. **Knowledge Kernel:** The multi-tier institutional memory (Transactional, Semantic, Relational, Temporal).
2. **Policy Kernel:** The deterministic engine compiling and executing business rules (JsonLogic, DMN, etc.).
3. **Decision Kernel:** The core producer of explainable, traversable Decision Graphs.
4. **Execution Kernel:** The workflow runtime (e.g., LangGraph, Temporal) handling scheduling, checkpointing, and interrupts.
5. **Governance Kernel:** The immutable ledger ensuring W3C PROV-O provenance, auditability, and human review chains.
6. **Simulation Kernel:** The counterfactual engine for stress testing and policy replay.
7. **Integration Kernel:** The sensory apparatus transforming external signals (Bloomberg, ERP) into standardized Events.

## 5. The Canonical Risk Ontology
All applications and kernels communicate via a strictly typed Canonical Risk Ontology (defined in `adam_os/core/ontology.py`). Nobody invents fields. The ontology enforces structures for Organizations, Financial Instruments, Legal Artifacts, and Risk Concepts.
