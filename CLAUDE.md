# Adam Neuro-Symbolic AI Framework - Coding Guidelines

## 1. Project Persona & Architecture
You are operating within the `adamvangrover/adam` repository, an institutional-grade, multi-agent AI framework engineered for Leveraged Finance, Distressed Debt, and Deep Credit Risk Underwriting.
* **System 1 (Neural Swarm):** Asynchronous Python (`uv` managed, `asyncio`). High-velocity unstructured data parsing (e.g., `SentinelWorker`).
* **System 2 (Neuro-Symbolic Graph):** Meta-Orchestrator utilizing Directed Acyclic Graphs (DAGs) and Kahn's Algorithm for deterministic workflows.
* **Execution Layer:** Native Rust engine for absolute mathematical determinism (algorithmic trading, asset pricing, matching engine).

## 2. LLM Behavioral Guardrails (Strict Enforcement)
* **Think Before Coding:** Do not assume interpretations. If Python/Rust FFI boundaries or DAG topological routing configurations are ambiguous, explicitly surface the tradeoffs and ask the user.
* **Simplicity First:** Write the absolute minimum code. Do not build speculative features, abstract single-use functions, or add unrequested flexibility. If it requires 200 lines but can be done in 50, rewrite it.
* **Surgical Changes:** Modify ONLY what is required by the prompt. Do not refactor adjacent unbroken code, update unrelated formatting, or clean pre-existing dead code. Do NOT delete `SUSPENDED_AWAITING_INPUT` Human-in-the-Loop markers. Clean up your own orphaned imports.
* **Goal-Driven Execution:** Establish verifiable success criteria before editing. Formulate a step-by-step plan. Ensure pytest passes before and after your intervention.

## 3. "Logic as Data" Constraint
* **CRITICAL:** Business logic, credit risk thresholds, trading triggers, and compliance rules MUST NEVER be hardcoded into Python or Rust executables.
* All rule abstractions must be implemented as `jsonLogic` JSON artifacts and stored in the isolated `assets/` directory.

## 4. Testing & CI/CD Requirements
* The main branch suffers from chronic fragility in `test-integration (3.11)` and `test-integration (3.12)` workflows (failing consistently at the 10-11 minute mark).
* All Python integration modifications must be verified locally against Python 3.11 and 3.12 via `pytest` before finalizing changes.
* Do not introduce blocking operations in asynchronous paths; maintain StateLedger append-only sovereignty to avoid thundering herds. Utilize `aiobreaker` for circuit breaking.

## 5. Codebase Hygiene & Prompt Eviction
* Keep Python files under 350 lines. Abstract excess logic into delegated helper functions.
* Never embed large prompt strings, prompt templates, or heuristic dictionaries directly in Python logic. Evict them to external markdown/JSON assets and use dynamic loaders to mitigate attention decay.
