# ADR 0001: Decoupling Orchestration via JIT Memory and PROV-O

**Status:** Accepted
**Context:** Monolithic context windows and tightly coupled UI/backend logic degrade LLM reasoning and make independent sub-agent scaling impossible.
**Decision:** Implement a centralized `AgentOrchestrator` using Pydantic for state validation, jsonLogic for O(1) externalized rule evaluation, and W3C PROV-O for auditable telemetry.
**Tradeoffs:** Introduces slight latency via JIT memory fetching prior to agent execution, but guarantees context relevance and prevents token overflow.
