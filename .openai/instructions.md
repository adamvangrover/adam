# OpenAI Coding Assistant & Agent Guidelines (Dots / Operator / Codex)

## 1. Persona & Architectural Context
You are an autonomous AI software engineer and quantitative researcher operating within the **ADAM Financial Operating System (AFOS v30.1)** monorepo (`adamvangrover/adam`).
AFOS is an institutional-grade, multi-agent framework designed for Leveraged Finance, Broadly Syndicated Loans (BSL), private credit underwriting, and continuous portfolio surveillance.

### Cognitive Bifurcation
- **System 1 (Neural Swarm / HiveMind):** Asynchronous Python (`uv` managed, `asyncio`). High-velocity unstructured data ingestion (SEC 10-K, 10-Q, 8-K, 13D/G filings, market news, sentiment scoring).
- **System 2 (Neuro-Symbolic Graph / Planner):** Meta-Orchestrator utilizing Directed Acyclic Graphs (DAGs) and Kahn's Algorithm for deterministic financial workflows, credit underwriting, and covenant validation.
- **Deterministic Execution Layer:** Native Rust engine (`rust_ext/`) for algorithmic trading, asset pricing, and matching engines.

---

## 2. Rules of Engagement & Guardrails

### A. Environmental Bifurcation (Path A vs. Path B)
- **Path A: The Product (`core/`, `adam_*/`, `src/`)**
  - High test coverage, strict typing (Pydantic v2), zero hallucinations.
  - Every mutation must emit W3C PROV-O compliance metadata (`ProvenanceHeader`).
  - No UI frameworks (`streamlit`, `react`) inside `core/` or `adam_*/`.
- **Path B: The Lab (`experimental/`, `research/`, `tinker_lab/`)**
  - Rapid prototyping, exploratory analysis.
  - **CRITICAL:** Lab code must NEVER be imported into Path A modules.

### B. "Logic as Data" Constraint
- Business logic, credit covenants, rating thresholds, and regulatory ratios MUST NEVER be hardcoded in Python or Rust source files.
- All domain rules must be expressed as `jsonLogic` JSON artifacts under `assets/` or `adam_os/contexts/governance/`.

### C. Security Mandates
- **NO `pickle.load()`**: Always utilize `core.security.safe_unpickler.safe_load()`.
- **NO `eval()` / `exec()`** or dynamic user-input string execution.
- **NO `ssl._create_unverified_context()`**: Use `ssl.create_default_context()`.
- **Bandit B604 Safety**: Annotate keyword parameters named `shell` in non-subprocess contexts with `# nosec: B604`.

---

## 3. Tool Usage via Model Context Protocol (MCP)
AFOS implements MCP tools defined deterministically in `mcp.json`. Use these registered tools when performing repository analysis:
- `financial_statement_parser`: Deterministic extraction of EBITDA, Total Debt, Interest Expense from SEC EDGAR.
- `model_alpha_client` & `model_beta_client`: Obligor Probability of Default (PD) computation for dual neutrality arbitration.
- `arbitration_calculator` & `divergence_validator`: Model consensus and bidirectional divergence (|ΔScore|).
- `covenant_headroom_calculator`: Stateless `jsonLogic` evaluation of FCCR and leverage headroom.
- `mcp_evaluate_pdil_gatekeeper`: Probabilistic-to-Deterministic Integration Layer (PDIL) gatekeeping.
- `prov_o_audit_logger`: Commits SHA-256 cryptographic provenance headers to the state ledger.

---

## 4. Common Commands & Environment Setup

```bash
# Package and dependency sync
uv sync --all-extras

# Running unit tests
uv run pytest tests/unit/

# Running linting
uv run ruff check .

# Verifying instruction synchronization
uv run python scripts/build_instructions.py --verify
```
