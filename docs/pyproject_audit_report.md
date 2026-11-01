# Architectural Audit Report: ADAM Financial Operating System (`adam-kernel` v30.1.0)

## Executive Summary

This architectural review documents the structural sanitation, modernization, and configuration audit of `pyproject.toml` and associated dependency management workflows for the **Autonomous Deterministic Alpha Matrix (ADAM) Financial Operating System** (`adam-kernel` v30.1.0).

The build and toolchain configuration directly supports ADAM's dual-path architectural doctrine:
* **Path A (Deterministic Production Core):** Strict Pydantic typing, zero-trust cryptographic validation, immutable audit trails, and deterministic Python/Rust PyO3 bindings.
* **Path B (Research, Exploration & Prototyping):** High-velocity financial experimentation, simulation sandboxes, and iterative backtesting harnesses.

This audit addresses five critical architectural categories to guarantee deterministic reproducibility across local developer environments and enterprise CI/CD pipelines.

---

## Category 1: Encoding & Unicode Parsing Sanitation

### 1.1 Root Cause & Manifestation
In previous iterations, non-breaking space (NBSP `\u00a0`) and zero-width characters accidentally entered configuration files (such as `pyproject.toml`) through copy-paste operations from web terminals, IDE rich text buffers, or formatted specification markdown. Standard Python TOML parsers (such as `tomllib` in Python 3.11+ and `tomli`) strictly reject non-standard whitespace outside of string literals, throwing unhandled syntax errors (`tomllib.TOMLDecodeError: Invalid character in number or identifier`) and halting CI pipelines before execution.

### 1.2 Sanitation & Remediation
* **Automated Stripping:** All NBSP bytes (`0xC2 0xA0`) and extraneous control characters were purged from `pyproject.toml`.
* **Parser Validation:** Validated programmatically via strict binary inspection and round-trip verification with Python 3.11's standard library `tomllib`:
  ```bash
  python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('TOML Validated')"
  ```
* **Git Hygiene:** Clean byte-level verification ensures cross-platform consistency between Linux CI runners, macOS ARM workstations, and Windows WSL2 environments.

---

## Category 2: Package Identity & Build Backend Harmonization

### 2.1 PEP 621 Metadata Alignment
Package metadata has been unified under PEP 621 specifications:
* **Package Name:** `adam-kernel`
* **Version:** `30.1.0`
* **Python Compatibility:** `>=3.11, <3.13` (guaranteeing stability with PyO3 0.22+ and Python 3.11/3.12 ABI compatibility).

### 2.2 Maturin & Rust PyO3 Integration
ADAM utilizes a hybrid deterministic execution model where high-throughput numerical and risk arbitration logic is compiled in Rust (`rust_ext/Cargo.toml`) and exposed via PyO3 bindings.
* **Build System:** `maturin>=1.0,<2.0` is configured as the PEP 517 build backend.
* **Package Discovery:** Explicit declaration of top-level Python namespaces in `tool.maturin.python-packages`:
  - `core` (System 2 deterministic engines, state managers, invariants)
  - `services` (gRPC and microservice adapters)
  - `src` (Zero-trust gateway, PDIL, agent coordination)
  - `adam_os` (Operating system primitives and task scheduler)
  - `adam_swarm` (Autonomous agent swarm orchestration)
  - `adam_finance` (Quantitative underwriting, EVE v2.0, pricing)
  - `adam_governance` (Fiduciary gatekeepers, jsonLogic validators)
  - `adam_graph` (Knowledge graph and SNC graph topology)

This explicit package list prevents Maturin from bundling ad-hoc directories or test sandboxes into the production wheel.

---

## Category 3: Modular Dependency Groups & Extras Topology

### 3.1 Core Deterministic Base
The base `project.dependencies` contains only dependencies necessary for boot, zero-trust verification, configuration loading, and basic orchestration:
* `pydantic>=2.11.10` & `pydantic-settings>=2.12.0`: Strict type validation and configuration schema enforcement.
* `python-dotenv>=1.2.2`, `uuid_utils>=0.12.0`: Runtime environment and UUID generation.
* `colorlog>=6.10.1`, `python-json-logger>=4.0.0`: Structured W3C PROV-O JSON telemetry logging.
* `json-logic-qubit`: Deterministic gatekeeper logic enforcement.
* `temporalio`: Distributed workflow and orchestration execution.

### 3.2 Granular Optional Extras
To eliminate monolithic dependency bloat in microservices and container deployments, dependencies are partitioned into targeted extras:
* `async-web`: Async communication and HTTP/WebSocket servers (`fastapi`, `uvicorn`, `aiohttp`, `websockets`, `Flask`).
* `data-science`: Numerical and statistical computing (`numpy`, `pandas`, `scipy`, `scikit-learn`, `sympy`, `statsmodels`, `optuna`).
* `ml-tensors`: Neural representations and local language models (`torch`, `transformers`, `sentence-transformers`, `triton`, `peft`).
* `orchestration`: Multi-agent synthesis and LLM frameworks (`openai`, `anthropic`, `litellm`, `langchain`, `langgraph`, `dspy-ai`, `pydantic-ai`).
* `finance`: Quantitative financial modeling and regulatory filings (`yfinance`, `ta`, `PyPortfolioOpt`, `edgartools`, `numpy-financial`, `weasyprint`, `markdown`).
* `quantum`: Quantum portfolio optimization stubs (`qiskit`, `qiskit-finance`, `qiskit-aer`).
* `infrastructure`: Enterprise storage, graph, and caching backends (`SQLAlchemy`, `neo4j`, `redis`, `celery`, `qdrant-client`).
* `observability`: Distributed tracing and telemetry (`opentelemetry-api`, `opentelemetry-sdk`, `mcp`, `fastmcp`).

### 3.3 Composite Profiles
* `full-kernel`: Aggregates the standard modular runtime for multi-node deployments.
* `risk-agent`: Specialized high-assurance underwriting bundle combining data science, financial modeling, and infrastructure.

---

## Category 4: Ruff Path A vs. Path B Governance & Granular `per-file-ignores`

### 4.1 Strict Path A Standards
Production code paths (`core/`, `src/`, `adam_os/`, `adam_finance/`, `adam_governance/`, `adam_graph/`) are held to rigorous standards:
* **Rule Sets:**
  - `E` / `F`: Pyflakes and core pycodestyle errors.
  - `I`: Deterministic isort import ordering with recognized first-party modules.
  - `B`: Flake8-bugbear defensive code patterns.
  - `ANN`: Strict type annotations (disallowing implicit `Any` and untyped parameters).
  - `UP`: Pyupgrade modern Python syntax (requiring `dict` over deprecated `typing.Dict`, modern union syntax).
  - `RUF`: Ruff-specific checks for common anti-patterns.
  - `C90`: McCabe complexity threshold bounded to `<= 10`.

### 4.2 Selective `per-file-ignores` Architecture
Because ADAM maintains a vast repository containing legacy scripts, test harnesses, prompt templates, and interactive mockups, blanket enforcement of rules like `ANN` (annotations) or `E501` (line length) on experimental code leads to developer friction or CI bypass workarounds.

`pyproject.toml` implements granular `per-file-ignores`:
* **`tests/**/*`:**
  - Ignore `ANN` (missing test parameter/return annotations), `S101` (allow `assert`), `B011`, and `E501`.
* **`scripts/**/*`, `ops/**/*`:**
  - Ignore `ANN` (ad-hoc CLI arguments), `C901` (procedural automation routines), and `E501`.
* **`showcase/**/*`, `verticals/**/*`, `mockups/**/*`:**
  - Ignore `ANN`, `E501`, `F401`, and `C901` for visualization demos and web prototypes.
* **`**/__init__.py`:**
  - Ignore `F401` (allow intentional package-level exports without warnings) and `F403`.

This bifurcation shields the Path A production kernel from code quality decay while allowing agile testing and research in Path B.

---

## Category 5: Automated CI/CD `uv.lock` Generation & Validation Strategy

### 5.1 Deterministic Verification with `uv sync --frozen`
The repository uses `uv.lock` to ensure byte-for-byte deterministic dependency installations across all environments.
* In GitHub Actions (`.github/workflows/ci.yml`), dependencies are installed strictly via:
  ```yaml
  - name: Verify Locked Environment
    run: uv sync --frozen --all-extras
  ```
  The `--frozen` flag guarantees that `uv` will fail fast if `pyproject.toml` has been altered without updating `uv.lock`.

### 5.2 Recommended Automated CI Validation Routine
To proactively catch lockfile drift and automate dependency synchronization, we recommend the following enhancements:

#### 1. Pre-Merge Lockfile Check in CI (`ci.yml`)
Add a dedicated step prior to dependency installation:
```yaml
- name: Validate Lockfile Freshness
  run: uv lock --check
```
`uv lock --check` verifies that `uv.lock` is up-to-date with respect to `pyproject.toml` without modifying the lockfile.

#### 2. Automated Lockfile Renewal Workflow (`.github/workflows/uv_lock_sync.yml`)
For branches updated through automated rituals or dependabot bumps, a dedicated workflow can maintain lockfile consistency:
```yaml
name: UV Lockfile Synchronization

on:
  pull_request:
    paths:
      - "pyproject.toml"

jobs:
  update-lockfile:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]' || github.event.pull_request.head.repo.full_name == github.repository
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.head_ref }}
      - uses: astral-sh/setup-uv@v5
        with:
          version: "0.5.24"
      - name: Relock Dependencies
        run: uv lock
      - name: Commit Updated Lockfile
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: update uv.lock to match pyproject.toml"
          file_pattern: "uv.lock"
```

---

## Conclusion & Readiness

The modernized `pyproject.toml` establishes a robust foundation for the ADAM Financial Operating System (`v30.1.0`), guaranteeing:
1. Complete elimination of Unicode and NBSP syntax parsing failures.
2. Direct alignment between PEP 621 package identity (`adam-kernel`) and the Maturin PyO3 build pipeline.
3. High-assurance protection of Path A core risk modules via tailored Ruff linting.
4. Total deterministic reproducibility via frozen `uv.lock` validation.
