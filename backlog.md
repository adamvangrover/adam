# Adam Platform Professionalization Backlog

## Current Status (PR: feat: professionalize adam architectural stack via IaC, JsonLogic governance, and provenance tracking)
This repository is currently transitioning from a research-focused prototype to a professional-grade platform. The current PR has laid the groundwork for deterministic architectural rigor and modularity.

### Accomplished in this Phase:
*   **Architectural Hardening:** Refactored `SecurityGovernanceGatekeeper` to include a `JsonLogicGovernanceGatekeeper` component (`src/pdil/middleware.py`). This decouples LLM probabilistic reasoning from deterministic risk engine executions. A legacy wrapper (`src/governance/gatekeeper.py`) delegates to the new logic evaluation check while retaining backwards compatibility.
*   **Infrastructure as Code (IaC):** Scaffolded initial Kubernetes manifests (`kubernetes/adam-os-kernel.yaml`, `kubernetes/sidecar-agents.yaml`) to deploy the core engine and swarm agents, moving away from isolated scripts.
*   **Observability & Provenance:** Laid the structural foundation for the `provenance_trace` component (`ProvenanceHeader`) in `AgentOutput` to ensure W3C PROV-O compliance. The mass-refactoring of `core/agents/` instantiations is deferred to the next phase to ensure stability.
*   **Diátaxis Framework Documentation:** Initialized the core structured documentation directories (`docs/tutorials/`, `docs/how-to/`, `docs/explanation/`, `docs/reference/`) alongside the root-level `SECURITY.md` and `CODE_OF_CONDUCT.md`.

## Future Session Instructions (Graceful & Additive Merging)
To close out this transition successfully, future sessions should focus on the following complementary tasks:

1.  **Iterative Integration of `jsonLogic`:** The current `JsonLogicGovernanceGatekeeper` provides the foundational interface. Future PRs should expand `jsonLogic` rule coverage across specific risk modules, ensuring they run additively alongside legacy heuristic checks.
2.  **Complete IaC Transition:** Expand the Kubernetes manifests to include persistent volumes, ingress routing, and config maps for secrets management, completing the transition from Docker Compose.
3.  **Dynamic Provenance Propagation:** Currently, `provenance_trace` is structurally required by `AgentOutput`. Future work must ensure that the generation of this header is dynamically linked to the exact data sources and jsonLogic evaluations used in the specific agent's execution context.
4.  **Populate Documentation:** Flesh out the Diátaxis placeholders with actual system specifications, ensuring the "Self-Healing" documentation mandate is met.
