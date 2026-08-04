# ADR 0001: Implementation of Deterministic Policy Engine via jsonLogic

## Status
Accepted

## Context
ADAM OS operates heavily within Temporal workflows, which strictly require deterministic execution. If a Temporal workflow encounters non-deterministic behavior (like floating point inconsistencies across architectures, random numbers, or external API mutations), history replay fails, causing catastrophic state corruption in financial simulations. We require a way to evaluate complex financial covenants (e.g., $LTV \le 0.35$) with zero risk of divergence.

## Decision
We will extract all financial covenant evaluation out of raw Python workflow code.
1. We will use **jsonLogic** to represent financial rules. jsonLogic provides an Abstract Syntax Tree (AST) represented entirely in JSON, ensuring that evaluation logic is mathematical, deterministic, and highly portable.
2. We will wrap this in a stateless `DeterministicPolicyEngine`.
3. Workflows will pass immutable `FinancialContext` state objects to Activities that invoke this engine.

## Alternatives Considered
*   **Python `eval()` or native expressions:** Rejected due to extreme security risks and lack of strict determinism guarantees.
*   **Drools/Java engines:** Rejected due to excessive JVM overhead and architectural misalignment with our Python/Temporal stack.
*   **Open Policy Agent (OPA) for mathematical logic:** While OPA is used for *authorization* in ADAM OS, writing complex quantitative financial math in Rego is unergonomic. We separate math (jsonLogic) from access control (OPA/Rego).

## Tradeoffs & Consequences
*   **Tradeoff:** Writing jsonLogic is slightly more cumbersome than writing native Python `if/else` statements.
*   **Consequence:** We achieve 100% replay-safety in Temporal workflows. Financial analysts can now update JSON files to tune parameters (like LTV thresholds) without altering core application code.

## Future Extensions
*   Build a UI to compile natural language covenants into jsonLogic payloads.
*   Integrate full gRPC OPA Sidecar calls replacing the current mock `OPAAuthorizationClient`.
