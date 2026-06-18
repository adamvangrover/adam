# Adam v3.0 Core Kernel

## Overview
This architectural shift transitions Adam v3.0 into an engine-agnostic, deterministic Core Kernel. By decoupling the raw cognitive capabilities (business logic, execution constraints, state validation) from the underlying LLM harness, the system remains completely modular.

## Milestone & MLOps Tracking
- **Phase 1: Foundation (Milestone 1):** Define core schemas (`kernel_rpc.json`, `logic_rules.json`, `prompt_matrix.jsonl`, `state_engine.json`). Implement the deterministic RPC server and basic routing.
- **Phase 2: Simulation & Logic:** Implement JSONLogic guardrails and local state transitions. Begin building adversarial loops.
- **Phase 3: Integration & Migration:** Shift legacy SEC Edgar scraping protocols, multi-agent tracking, and quantitative risk simulations to use the new `/kernel` API.
- **CI/CD & MLOps Integration:** Integrate CI/CD for schema validation using `jsonschema` and JSONLogic testing during pull requests. Use GitHub Actions to enforce syntax and structural validity before deployment. Weave will trace System 1/System 2 boundaries.

## Token Usage & Efficiency
- Schemas are designed to be aggressively token-efficient.
- Redundant keys are stripped in favor of compact, predictable JSON structures.
- JSON-RPC 2.0 ensures lightweight protocol overhead optimized for LLM context windows.

## Model Selection Routing Strategy
The harness routes specific cognitive tasks to the best-suited model family based on the current JSONL state:
- **Gemini:** Routed for high-context synthesis and multi-document reasoning.
- **Claude:** Routed for meticulous formatting and deep analytical writing.
- **GPT-4o:** Routed for rapid tool calling, tool use, and complex function execution.

## Isolated Sandbox Plan
- **Local-first Execution:** The state engine and deterministic execution layer run entirely locally, zero-dependency.
- **Separation of Concerns:** Complete separation between the stateless LLM calls (API/network layer) and the stateful, private JSON runtime execution (local sandbox). State sovereignty via immutable-append-only `StateLedger`.
