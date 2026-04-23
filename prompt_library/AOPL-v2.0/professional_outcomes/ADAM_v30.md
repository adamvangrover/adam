Here is the unified, master iteration of the **ADAM Protocol v30.0 Remediation & Architecture Plan**. This version seamlessly integrates your foundational governance rules with the recent architectural audit findings, completed remediation phases, and the strategic roadmap to core autonomy. 

This document is formatted to serve as the definitive architectural manifesto for your repository.

***

# ADAM Protocol v30.0: Neuro-Symbolic Financial Sovereign
## Unified Architecture, Remediation, & Strategic Roadmap

## Executive Summary
This document outlines the architectural doctrine and ongoing strategic roadmap for the ADAM ecosystem. Following a rigorous architectural audit, the repository has undergone a 5-phase structural remediation to eliminate artifact pollution, fragment the dependency graph securely, and solidify the Rust execution layer. Moving forward, the protocol enforces strict dependency enclaves, immutable telemetry management, and a deliberate boundary between deterministic execution and multi-agent neural reasoning.

---

## PART I: Architectural Standardization & Governance

### 1. Strategic Dependency Enclaves (Workspace Architecture)
Forcing a monolithic dependency graph across the entire multi-agent ecosystem introduces severe risk. Heavy NLP/OCR libraries required for SEC EDGAR triage will not be allowed to bloat the deterministic environments required for core risk models.
* **The Global Core:** A root-level `pyproject.toml` maintains *only* universally required, highly stable packages (e.g., Pydantic, standard logging).
* **Domain-Specific Manifests:** Specialized microservices (e.g., the WhaleScanner intelligence core) maintain explicit dependency manifests, remaining effectively air-gapped from execution-critical trading logic.
* **Workspace Execution:** `uv` is configured to manage the multi-project workspace, ensuring agents run exclusively with the specific fragments of the dependency tree they require.

### 2. Deployment Redundancy
Isolated, redundant environments are defined via centralized configuration to prevent systemic failure.
* All deployment Dockerfiles are consolidated in `docker/`.
* `docker-compose.yml` explicitly routes to `Dockerfile.core` for pure financial processing and `Dockerfile.swarm` for asynchronous multi-agent coordination.

### 3. State Governance & The Immutable Intelligence Ledger
The root directory is pristine. Iterative fixes (`patch_*.py`, `verify_*.py`) live strictly on temporary feature branches and are pruned upon merge. Operational scaffolding lives in `scripts/ops/`, and raw Python entry points are isolated in `src/`.
* **The Authorized Hierarchy:**
    * `/telemetry_history/`: The formal ledger for deprecated states. Code is not deleted; it is tagged with `@deprecated`. Logic and deprecation rationales are wrapped in standardized JSONL formats here as queryable context for future prompt engineering.
    * `/config/mocks/`: The isolated environment for synthetic data stubs.
    * `/core/`: The pristine, deterministic core logic for the System 1/System 2 swarms.
    * `/scripts/ops/`: Verified, production-ready operational wrappers (e.g., `orchestrator.py`).
    * `/docker/`: Consolidated containerization logic.
* **`.gitignore` Enforcement:** Explicitly bans `*.log`, `*.egg-info/`, `.Jules/`, and temporary `verify_*.py` files.

---

## PART II: Execution Dynamics & Mock Isolation

### The `MOCK_MODE` Protocol
Pervasive mocking in core workflows (e.g., `MockLLM`, `_build_mock_graph`) was identified as a primary architectural bottleneck. Mocking is no longer treated as technical debt, but rather as a formalized, strictly isolated fallback mechanism.
* **Fail-Fast Production:** The system enforces an explicit `MOCK_MODE=true` environment variable check. In production (`MOCK_MODE=false`), the system will aggressively fail fast via `ImportError` or `ValueError` if real connections (e.g., IBKR, Neo4j) drop.
* **Context Preservation:** When `MOCK_MODE=true` (demo/sandbox environments), isolated mock files in `/config/mocks/` take over to ensure stable frontend regressions and uninterrupted UI iteration.

---

## PART III: Agentic Expansion & The Intelligence Swarm (AOPL)

### Telemetry & Prompt Engineering
Deprecated logic is actively synthesized into training data for future architectural builds via `swarm_telemetry_expanded.jsonl`.
* **Inheritance Limits:** Agents in `v30_architecture/python_intelligence` must inherit from `BaseAgent` and strictly utilize the `NeuralMesh` bridge (`emit_packet()`).
* **kwargs Requirement:** All agent `execute()` methods must accept `**kwargs` to prevent async and multi-threading validation errors.
* **Pydantic Alignment:** TypeScript models must strictly mirror Python Pydantic definitions (e.g., `CreditMetrics`) to prevent silent `422` validation failures.
* **The AOPL:** The Adam Operational Prompt Library (`prompt_library/`) dictates markdown-based prompt templates. Hard-coding long heuristics into class instances is forbidden.
* **LLM as Judge:** Stochastic outputs must pass through the dual-layered evaluation harness (`evals/eval_illiquid_market.py`) for deterministic logic gate checks followed by the `ConvictionScorer`.

---

## PART IV: Strategic Roadmap to Core Autonomy (Next Steps)

To transition from a structurally sound repository to a fully autonomous financial sovereign, the following roadmap is active:

### 1. Data Ingestion Hardening
* **Objective:** Completely deprecate `MOCK_MODE` stubs for internal data generation.
* **Execution:** * Implement full SEC EDGAR fetching via `edgartools` inside `SpreadingEngine` (replacing hardcoded AAPL/TSLA arrays).
    * Connect `GraphEngine` to a local Dockerized `Neo4j` instance, mapping relationships dynamically via Cypher queries.

### 2. The PyO3 Rust/Python Bridge
* **Objective:** Eradicate the fragmented execution environment by unifying the System 1 Python Swarm with the System 2 Rust Execution Layer.
* **Execution:** * Extend `core/rust_pricing/src/lib.rs` PyO3 bindings to expose the `BTreeMap`-backed `OrderBook` and the `RiskEngine` (tracking drawdown and inventory).
    * Implement an asynchronous message queue (Redis/NATS JetStream) to stream Python-derived sentiment signals directly into the Rust `RiskEngine` to dynamically adjust the Avellaneda-Stoikov spread parameters.

### 3. LangGraph & LiteLLM Refactoring
* **Objective:** Eliminate `MockLLM` overrides in standard execution.
* **Execution:** * Refactor LangGraph nodes to strictly rely on LiteLLM for robust multi-provider routing (OpenAI, Anthropic) with exponential backoff.
    * Implement the System 2 DAG to enforce strict deterministic rules (utilizing `jsonLogic`) prior to returning final outputs to the Unified Ledger.

---

## PART V: Execution Directives for Continued Remediation

If further optimization is required, execute the following specific prompts:

**Prompt 1: Security Audit & Pre-Commit Linting (Workspace Aware)**
> Run `uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` across all configured workspaces to verify standard syntax compliance. Follow it by running `uv run bandit -r core services scripts -ll -x tests`. Do not suppress valid vulnerabilities with `# nosec`. Resolve the architectural flaw. Ensure `PYTHONPATH=.` is prefixed before testing local modules within their respective enclaves.

**Prompt 2: V30 Architecture Expansion**
> Under the guidelines of 'Protocol ARCHITECT_INFINITE', create a new specialized agent derived from `BaseAgent` inside `core/v30_architecture/python_intelligence/agents/`. Ensure the `execute(**kwargs)` signature is respected. Register the agent inside the `swarm_runner.py` configuration array and write a comprehensive test targeting only the new agent file using `uv run pytest`.

**Prompt 3: Component Deprecation & Telemetry Logging**
> Audit the `core/agents/` (legacy directory) and `core/v30_architecture/python_intelligence/agents/`. Map any overlapping logic. Apply the `@deprecated` decorator to legacy agents. Extract their core logic loops and docstrings, format them as JSON, and append them to `/telemetry_history/swarm_telemetry_expanded.jsonl`. Transition workflows to the v30 framework while allowing legacy agents to gracefully log warnings.
