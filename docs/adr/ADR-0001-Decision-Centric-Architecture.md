# ADR-0001-Decision-Centric-Architecture

## Context
The system has reached an inflection point where the architecture is no longer best described as a "multi-agent risk framework." It is closer to a Financial Decision Operating System (FDOS)—a domain-specific operating system for institutional financial decision-making. The current architecture is organized around agents, which should become implementation details rather than architectural primitives. The system needs to be reorganized around seven foundational kernels.

## Decision
We are evolving the platform from a workflow-centric architecture to a decision-centric architecture named the Adam Financial Operating System (AFOS). The core abstractions will be evidence, policies, events, and decisions, supported by seven foundational kernels:
1. **Knowledge Kernel:** A three-tier memory architecture (Transactional, Semantic, Relational, Temporal).
2. **Policy Kernel:** A full policy engine that parses, compiles, and executes various policy DSLs.
3. **Decision Kernel:** Produces explainable decision graphs rather than just final risk ratings.
4. **Execution Kernel:** Manages workflows and agent runtimes as replaceable implementation details.
5. **Governance Kernel:** Records provenance and audit ledgers, enabling replayable decisions.
6. **Simulation Kernel:** Enables historical portfolio replays and macro shock simulations.
7. **Integration Kernel:** An event-driven bus for all external and internal data flows.

Additionally, a Canonical Risk Ontology will be established to provide a shared language (Organization, Financial Instrument, Legal Artifact, etc.) across all services.

## Alternatives
- Maintain the current "multi-agent risk framework": Rejected because agents are execution engines, not the foundational architecture, and a workflow-centric model is less durable than a decision-centric one.

## Tradeoffs
- Requires significant refactoring of existing workflows to adopt the decision-first model.
- Increases initial complexity by introducing a strict ontology and distinct kernels.
- Provides long-term durability, testability, and reuse across multiple financial applications.

## Consequences
- Policies become independently testable and versioned.
- Decision provenance is complete even if orchestration changes.
- Multiple workflows (underwriting, annual review, watchlist monitoring, stress testing) can reuse the same decision logic.
- Regulatory audits can focus on decisions and evidence rather than execution paths.
- LLMs are treated strictly as replaceable execution engines that generate evidence, not architectural primitives.

## Future Extensions
- Expand the Policy Kernel compiler to natively support DMN, SQL predicates, YAML policies, and regulatory policies.
- Implement full alternative scenario macro shock testing within the Simulation Kernel.
