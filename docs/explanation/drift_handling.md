# Explanation: System Drift and State Recovery

This document explains how the ADAM architecture monitors, detects, and manages state drift between probabilistic inference components (System 1) and deterministic mathematical execution engines (System 2 Neuro-Symbolic Graph).

## The State Drift Problem
Stochastic models exhibit output variance and behavioral drift over time. In high-stakes institutional credit underwriting, uncontrolled variance is unacceptable.

In neuro-symbolic systems, **drift** occurs when the outputs of probabilistic components (e.g., extracted metrics from SEC filings) diverge from the strict mathematical bounds expected by deterministic execution engines (e.g., VaR calculators, matching engines). Over time, as external data distributions shift or inference layers experience statistical deviation, this divergence can compromise system reliability.

## Drift Handling Mechanisms: Detection via `observed_drift` Flags
To mitigate variance, ADAM employs a rigorous drift detection and state recovery mechanism managed by the `DriftIntelligenceLayer`.

ADAM utilizes continuous provenance tracing and boundary validation via the `GovernanceGatekeeper`. When a probabilistic component outputs a Pydantic schema, the governance layer cross-references these values against historical baselines and hard-coded mathematical bounds.

When the orchestrator processes data between deterministic models and probabilistic components, it triggers **revalidation workflows**. If a discrepancy is identified (e.g., extracted EBITDA volatility exceeds defined thresholds without a corresponding corporate event), or if an execution anomaly is detected (e.g., a probabilistic output contradicts the deterministic JSONLogic covenants), the orchestrator intercepts the process and flags the state with an **`observed_drift = True`** boolean.

These flags prevent the downstream propagation of invalid state. The anomaly is recorded in the `DriftStorageBackend`.

## Revalidation Workflows (State Recovery)
Once `observed_drift` is triggered, the system initiates an automated state recovery workflow:

1. **Pause Downstream Execution**: Execution layers (e.g., Rust matching engines) are immediately halted for the affected task pipeline to prevent state corruption.
2. **Re-grounding**: The probabilistic layer is instructed to re-run the inference using a fallback parser or an alternative deterministic heuristic to verify the primary extraction. The system may fallback to `IndependentGatekeeperCheck` or engage `CircuitBreaker` redundancy.
3. **Human-in-the-Loop Gate**: If the drift cannot be autonomously resolved through heuristic consensus or fallback parameters, the task state is set to `PENDING` and elevated to the `resolve_human_gate` for manual review by a quantitative analyst.
4. **Resumption**: Once the parameters are corrected or manually overridden, the `observed_drift` flag is cleared, and the DAG execution resumes deterministically.

This state recovery loop ensures that the deterministic execution layer remains uncorrupted by stochastic variance, guaranteeing that anomalies never cascade into financial execution.
