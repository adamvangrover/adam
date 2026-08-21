# ADR-0004-Harness-Telemetry-and-Managed-Loops

## Context
As defined in ADR-0003, horizon swarms operate continuously across an asynchronous environment. The lack of standard, discrete request-response boundaries means traditional logging is insufficient. We need to construct a comprehensive observability framework capable of monitoring agentic health, detecting behavioral drift, and capturing semantic reasoning trails over extended periods.

## Decision
We will implement "Harness Telemetry and Managed Loops," moving away from passive logging to an active, structural monitoring paradigm.

1.  **Managed Execution Loops:**
    *   Agents will not run unstructured `while True` loops. Instead, they will operate within a "Managed Loop Harness."
    *   This harness enforces heartbeat intervals, resource quotas, and deterministic checkpointing.
    *   At the end of each managed cycle, the agent's internal state must be serialized into a W3C PROV-O compliant format before yielding to the orchestrator.

2.  **Harness Telemetry Infrastructure:**
    *   We will introduce a decentralized telemetry bus. Instead of agents writing directly to a centralized log, they emit structured events (`SwarmPulseEvent`, `ReasoningTraceEvent`, `AnomalyEvent`) onto this bus.
    *   Telemetry will capture not just the "what" (actions taken) but the "why" (probabilistic distribution of choices considered, attention weights on context, confidence scores).
    *   This data will be routed to a time-series specialized storage and indexed against the Universal Knowledge Graph to allow for temporal querying of the swarm's thought process.

## Status
Accepted

## Consequences
- **Observability:** Provides unprecedented insight into long-running agent behavior, allowing operators to visualize the evolving strategy of the swarm in real-time.
- **Intervention:** The Managed Loop structure allows the Meta-Orchestrator to pause, rewind, or forcefully adjust an agent's context window if telemetry indicates a deviation from deterministic guardrails.
- **Performance Overhead:** Emitting rich, structured telemetry continuously will consume compute and bandwidth. Optimization of serialization (e.g., using Rust-backed binary formats for the pulse events) will be required to maintain the latency requirements of the execution layer.
