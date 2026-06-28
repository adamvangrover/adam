# Explanation: Architecture and Drift Handling

ADAM utilizes a Tri-Layer architecture designed specifically for institutional finance. The core tenet is maintaining a deterministic execution layer alongside a probabilistic intelligence swarm.

## The Epistemological Crisis
Probabilistic models (System 1) suffer from hallucinations and behavioral drift over time. In high-stakes credit underwriting, this is unacceptable.

## Self-Healing Documentation and Drift Handling
To combat this, ADAM employs a rigorous drift detection and self-healing mechanism managed by the `DriftIntelligenceLayer`.

When the system handles drift between deterministic models and probabilistic AI agents, it triggers **revalidation workflows**. If an anomaly or deviation is detected during execution (e.g., an LLM outputs a covenant violation interpretation that contradicts the deterministic JSONLogic), the system intercepts the process and sets **`observed_drift` flags**.

These flags prevent downstream propagation of the error. The anomaly is recorded in the `DriftStorageBackend`, and the system may fallback to `IndependentGatekeeperCheck` or engage `CircuitBreaker` redundancy. This ensures that the execution layer remains uncorrupted by probabilistic hallucinations, achieving true Self-Healing autonomy.
