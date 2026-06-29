# The Self-Healing Architecture

The Adam Platform operates at the intersection of probabilistic AI (Large Language Models) and deterministic execution (Rust engines, jsonLogic). To bridge this gap safely, the platform employs a **Self-Healing Architecture**.

## The Problem: Epistemological Drift

Stochastic models hallucinate. In institutional finance, a hallucinated calculation is a catastrophic failure. When an agent's reasoning drifts from established mathematical reality, we call this **Epistemological Drift**.

## The Solution: Additive Governance and Healing

To combat drift, Adam utilizes the `JsonLogicGovernanceGatekeeper`.

1.  **Evaluation:** Before and after an agent executes, its input and output payloads are evaluated against strict `jsonLogic` schemas.
2.  **Observation:** If a constraint is violated (e.g., `ebit < 0`), the execution is not immediately halted. Instead, an `observed_drift` flag is set to `True`.
3.  **Healing:** The `DriftIntelligenceLayer` intercepts the flagged payload, triggering a revalidation workflow. The agent is prompted to correct its reasoning, or the system falls back to a deterministic calculation.

## W3C PROV-O Compliance

Every decision made by the platform is tracked using a `ProvenanceHeader`. This trace includes a cryptographic hash of the content (`content_hash`), the exact time of execution, and the specific version of the logic rules used. This ensures an immutable audit trail for every inference.
