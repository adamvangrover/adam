# Institutional-Grade Neuro-Symbolic Financial Sovereign: Remediation & Architecture Plan

## Executive Summary
This document provides a comprehensive post-mortem and remediation strategy based on the enforcement of strategic dependency enclaves, immutable telemetry management, formalization of mock execution layers, and integration of experimental time-series models. The goal is to enforce a strictly modular, redundant, and secure institutional-grade architecture for the ADAM ecosystem while preserving the historical context of iterative development.

---

## 1. Architectural Standardization & Setup

### Strategic Dependency Enclaves (Workspace Architecture)
- **The Issue:** Forcing a monolithic dependency graph across the entire multi-agent ecosystem introduces risk. Heavy NLP and OCR libraries required for SEC EDGAR triage can bloat or destabilize the lightweight, deterministic environments required for core risk models.
- **The Solution:** We have adopted a controlled fragmentation approach using a **Workspace Architecture** managed via `uv`.
- **Action Items:**
  - **The Global Core:** Maintain a root-level `pyproject.toml` containing *only* universally required, highly stable packages (e.g., Pydantic, standard logging utilities).
  - **Domain-Specific Manifests:** Specialized microservices (e.g., WhaleScanner intelligence core) must maintain their own explicit dependency manifests to remain effectively air-gapped from execution-critical trading logic.
  - **Workspace Execution:** Configure `uv` to manage the multi-project workspace, ensuring agents run exclusively with the specific fragments of the dependency tree they require.

### Docker Redundancy
- **The Issue:** Attempting to force a complex multi-agent system into a single container image limits flexibility and deployment targets.
- **The Solution:** Leverage isolated, redundant environments defined via `docker-compose.yml`.
- **Action Items:**
  - Ensure the `docker-compose.yml` explicitly routes to `Dockerfile.core` for pure financial processing.
  - Ensure `Dockerfile.swarm` is used exclusively when asynchronous multi-agent coordination is required.
  - Retain `Dockerfile.modern` for future iterative API updates without disrupting the legacy core.

---

## 2. Directory Hierarchy & State Governance

The root directory must remain pristine. No logs, verification dumps, or temporary patch scripts are permitted in the main file tree. We treat legacy code and UI states as an **Immutable Intelligence Ledger**.

### Component Lifecycle & Branch-Level Pruning
- **Git Hygiene over File Graveyards:** Iterative fixes (`patch_*.py`, `verify_*.py`) must live strictly on temporary feature branches. Once verified and merged into the deterministic core, the branch is closed.
- **The `@deprecated` Protocol:** Instead of abruptly moving old agents, tag them with a standard Python `@deprecated` decorator. This allows graceful degradation or logging warnings in mock environments during the v30 transition.

### The Authorized Hierarchy
- **`/telemetry_history/`**: The formal ledger for deprecated states. When a module is refactored out of the active core, its final state, logic, and deprecation rationale are wrapped in standardized JSONL formats here, acting as queryable context for future prompt engineering.
- **`/config/mocks/`**: The formal "Static Mock Mode" (see Section 3).
- **`/experimental/`**: Time-series ML experiments, Monte Carlo benchmarks, and iterative predictive modeling (e.g., `benchmark_rng.py`, `ml_forward_outlook.py`). These provide valuable context but should not pollute core operational flow.
- **`/core/`**: The pristine, deterministic core logic for the System 1/System 2 swarms.
- **`/scripts/`**: Verified, production-ready operational wrappers (e.g., `daily_ritual.py`, `generate_comprehensive_index.py`).
- **`/showcase/`**: The static HTML artifacts for dashboards, explicitly skipping `index.html` via regex parsing.
- **`.gitignore` Enforcement:**
  - Explicitly ban `*.log`, `*.egg-info/`, `.Jules/`, and temporary `verification_*.py` files from the main branch.

---

## 3. Formalized "Static Mock Mode"

### The Rationale
"Mocking" in an institutional neuro-symbolic engine is not technical debt—it is a critical fallback mechanism. When live data pipelines fail or the Rust execution layer requires compilation, the frontend and UI iteration loops must survive independently.

### The Implementation
- Mock files (`mock_edgar.py`, `mock_llm_generator.py`) have been moved out of the pristine `core/pipelines/` and `core/engine/` directories into `config/mocks/`.
- **Routing:** Application layers (e.g., `ICATEngine`) dynamically route logic based on the environment flag.
  ```python
  import os
  if os.getenv("ENV") == "demo":
      return self._mock_edgar_fetch_icat(ticker)
  else:
      return self._live_rust_execution_fetch(ticker)
  ```
- **Context Preservation:** Simulated logic triggers, like the deterministic synthetic credit memos generated by `LiveMockEngine`, ensure stable frontend regressions while disconnected from Live IBKR/Alpaca feeds.

---

## 4. Agent Expansion & Prompt Engineering (AOPL)

### Telemetry Context
Deprecated logic is actively synthesized into training data. The logic, architecture decisions, and docstrings of phased-out components are parsed and appended to `swarm_telemetry_expanded.jsonl`. This ensures that future LLM-based architectural builds have the contextual history of past refactors and iterative UI states.

### Workflow Rules
1. **Inheritance Limits:** When building agents in the `v30_architecture/python_intelligence`, inherit from `BaseAgent` and strictly utilize the `NeuralMesh` bridge (`emit_packet()`).
2. **kwargs Requirement:** All agent `execute()` methods must accept `**kwargs` to prevent async and multi-threading validation errors.
3. **Pydantic Alignment:** Ensure that TypeScript models strictly mirror the Python Pydantic definitions (e.g., `CreditMetrics`) to prevent silent validation failures (like `422 Missing Field`).
4. **Prompt Templates:** Leverage the Markdown-based Adam Operational Prompt Library (AOPL) located in `prompt_library/`. Avoid hard-coding long heuristic prompt blocks into the agent class instances directly.
5. **LLM as Judge:** Utilize the dual-layered evaluation harness (`evals/eval_illiquid_market.py`) for stochastic outputs, wrapping them in a deterministic logic gate check followed by the `ConvictionScorer`.

---

## 5. Execution Prompts for Continued Remediation

If further optimization is required, execute the following prompt sequences to maintain standards:

**Prompt 1: Security Audit & Pre-Commit Linting (Workspace Aware)**
```text
Run `uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` across all configured workspaces to verify standard syntax compliance. Follow it by running `uv run bandit -r core services scripts -ll -x tests`. Do not suppress valid vulnerabilities with `# nosec`. Resolve the architectural flaw. Ensure `PYTHONPATH=.` is prefixed before testing local modules within their respective enclaves.
```

**Prompt 2: V30 Architecture Expansion**
```text
Under the guidelines of 'Protocol ARCHITECT_INFINITE', create a new specialized agent derived from `BaseAgent` inside `core/v30_architecture/python_intelligence/agents/`. Ensure the `execute(**kwargs)` signature is respected. Register the agent inside the `swarm_runner.py` configuration array and write a comprehensive test targeting only the new agent file using `uv run pytest`.
```

**Prompt 3: Component Deprecation & Telemetry Logging**
```text
Audit the `core/agents/` (legacy directory) and `core/v30_architecture/python_intelligence/agents/`. Map any overlapping logic. Instead of deleting legacy agents, apply the `@deprecated` decorator. Extract their core logic loops and docstrings, format them as JSON, and append them to `/telemetry_history/swarm_telemetry_expanded.jsonl`. Generate a plan to transition workflows to the v30 framework while allowing legacy agents to gracefully log warnings during the transition.
```
