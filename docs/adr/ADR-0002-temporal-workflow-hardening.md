# ADR 0002: Temporal Workflow Hardening

**Status:** Accepted
**Context:** Agent executions involving external LLMs or complex financial risk calculations are inherently non-deterministic and prone to network failures. Executing these directly in the orchestrator violates requirements for replayability and auditable fault tolerance.
**Decision:** Migrate the execution payload out of the synchronous `AgentOrchestrator` and into a Temporal `AgentExecutionWorkflow`. All side-effects (LLM inference, database writes) are wrapped in strictly typed Temporal Activities (`execute_agent_inference`).
**Tradeoffs:** Introduces dependency on a Temporal cluster and adds workflow boilerplate, but guarantees event sourcing, automatic retries, and time-travel debugging for complex financial transactions.
