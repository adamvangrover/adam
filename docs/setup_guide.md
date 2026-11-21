# ADAM Setup Guide

One guide, many setups. ADAM is modular: you can run a single slice (just the orchestrator, just the
credit math, just a Rust kernel, just the static UI) or the full Docker stack. Pick the slice you need.

> Consolidates the former `docs/archive/SETUP_GUIDE_v23.md`, [`alternative_setup.md`](alternative_setup.md),
> [`custom_builds.md`](custom_builds.md) and [`deployment.md`](deployment.md). Those files remain for
> detail; where they disagree with this guide, this guide reflects the current tree.

Related: [README](../README.md) · [REPO_MAP](../REPO_MAP.md) · [ARCHITECTURE](ARCHITECTURE.md) ·
[AGENTS.md](../AGENTS.md) · [TEMPLATES](TEMPLATES.md)

## Contents

- [Choose a slice](#choose-a-slice)
- [Prerequisites](#prerequisites)
- [Common base: uv + Python 3.12](#common-base-uv--python-312)
- [Environment variables](#environment-variables)
- [Slice A — Static showcase only](#slice-a--static-showcase-only-no-install)
- [Slice B — Minimal: orchestrator only](#slice-b--minimal-orchestrator-only)
- [Slice C — credit_sentinel only](#slice-c--credit_sentinel-only)
- [Slice D — System 1 swarm](#slice-d--system-1-swarm-neuralmesh)
- [Slice E — Rust slices](#slice-e--rust-slices)
- [Slice F — Frontend](#slice-f--frontend)
- [Slice G — Docker / full stack](#slice-g--docker--full-stack)
- [Slice H — Custom bundles](#slice-h--custom-bundles)
- [Slice I — MCP server](#slice-i--mcp-server)
- [Cloud & Kubernetes](#cloud--kubernetes)
- [Verify your install](#verify-your-install)
- [Troubleshooting](#troubleshooting)

## Choose a slice

| Slice | You get | Needs | Time |
|---|---|---|---|
| A. Static showcase | Dashboards, terminals, newsletters in a browser | A browser | 0 min |
| B. Orchestrator only | `MetaOrchestrator.route_request`, graphs, evals | uv, Python 3.12 | ~5 min |
| C. credit_sentinel only | Ratio calculator + distress classifier | Python, `numpy` (`scikit-learn` optional) | ~2 min |
| D. System 1 swarm | NeuralMesh agents + WebSocket bridge (ports 8000/8001) | Slice B + `async-web` extra | ~5 min |
| E. Rust slices | Deterministic kernels, PyO3 extensions | Rust toolchain (`cargo`), `maturin` for Python bindings | ~5–15 min |
| F. Frontend | React client (`services/webapp/client/`) or Streamlit UIs | Node + pnpm/npm, or `streamlit` | ~5 min |
| G. Docker / full stack | API, core engine, client, Postgres, TimescaleDB, Qdrant, Neo4j, Redis, worker | Docker + Compose | ~10–20 min |
| H. Custom bundles | Exported Lite/Standard/Full subset of modules | Python | ~2 min |
| I. MCP server | Tools from `mcp.json` for MCP clients | Slice B | ~2 min |

## Prerequisites

| Requirement | Detail |
|---|---|
| OS | Linux, macOS, or Windows via WSL2 |
| Python | 3.12 recommended; `pyproject.toml` allows `>=3.11,<3.13` |
| Package manager | [`uv`](https://astral.sh/uv) |
| Optional | Docker + Compose (G), Rust/cargo (E), Node 18+ with pnpm or npm (F) |

## Common base: uv + Python 3.12

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
git clone https://github.com/adamvangrover/adam.git && cd adam
uv venv --python 3.12                             # creates .venv/
```

Dependencies are split into extras in `pyproject.toml`:

| Extra | Contents (abridged) | Use for |
|---|---|---|
| *(base)* | pydantic, pydantic-settings, python-dotenv, json-logic-qubit, temporalio | Contracts only |
| `data-science` | numpy, pandas, scipy, scikit-learn, statsmodels | Models, credit math |
| `orchestration` | openai, anthropic, litellm, langchain, langgraph, dspy, pydantic-ai | Orchestrator + graphs |
| `async-web` | fastapi, uvicorn, Flask (+CORS/SocketIO/JWT), aiohttp, websockets | API, swarm bridge |
| `finance` | yfinance, ta, PyPortfolioOpt, edgartools, weasyprint | Market data, PDF reports |
| `ml-tensors` | torch, transformers, sentence-transformers | Heavy ML (optional) |
| `quantum` | qiskit, … | Quantum experiments (mocked in CI) |
| `full-kernel`, `risk-agent`, `infrastructure`, `observability`, `port-core` | See `pyproject.toml` | Specialized slices |

> **Pin conflict:** `requirements/base.txt` contains conflicting pins (opentelemetry). Prefer
> `uv sync --extra ...` over `pip install -r requirements/base.txt`, and if an import still fails,
> install that package individually: `uv pip install <package>`.

## Environment variables

```bash
cp .env.example .env
```

All keys are optional. Without an LLM key the orchestrator logs a warning and uses mock behaviour
(e.g. `GEMINI_API_KEY not found. Defaulting to Mock behavior`). Typical keys:

```ini
GEMINI_API_KEY=...        # default orchestrator model family (gemini-4-neon)
OPENAI_API_KEY=...        # optional alternative providers
ANTHROPIC_API_KEY=...
FMP_API_KEY=...           # market data
POSTGRES_URI=postgresql://user:pass@localhost:5432/adam
REDIS_URL=redis://localhost:6379/0
```

The orchestrator model defaults to `gemini-4-neon` (`core/engine/meta_orchestrator.py::OrchestratorConfig`)
and is overridden by `routing_tiers.tier_1_primary_system2.model_id` in `config/llm_routing.yaml`.
Models listed in `core/config/models.py::BLOCKED_MODELS` are refused at startup (SR 11-7).

---

## Slice A — Static showcase only (no install)

Open in a browser:

- [`index.html`](../index.html) — system portal
- [`showcase/index.html`](../showcase/index.html) — standalone credit sensitivity engine
- [`showcase/dashboards/index.html`](../showcase/dashboards/index.html) — dashboard catalog (from `manifest.json`)

Or serve locally: `python3 -m http.server 8080` and open `http://localhost:8080/`.
Template families are described in [TEMPLATES.md](TEMPLATES.md) and [showcase/README.md](../showcase/README.md).

## Slice B — Minimal: orchestrator only

```bash
uv sync --extra data-science --extra orchestration
```

```python
import asyncio
from core.engine.meta_orchestrator import MetaOrchestrator

orch = MetaOrchestrator()
print(asyncio.run(orch.route_request("Simulate a market crash", context={"portfolio": {"aum": 1e6, "allocation": {"SPY": 0.6, "TLT": 0.4}}})))
```

Queries without a resolvable ticker/entity return `{"status": "clarification_needed"}`. Check routing
accuracy without any API key:

```bash
.venv/bin/python tests/evals/run_accuracy.py   # prints query_route / query_entity vs. 90% floors
```

CLI entry point: `uv run python scripts/run_adam.py` (wraps `core/main.py`).

## Slice C — credit_sentinel only

The deterministic credit math in `core/credit_sentinel/` imports with only `numpy` (plus
`scikit-learn` for the Random Forest; without it the classifier uses a heuristic fallback).

```bash
uv pip install pydantic numpy scikit-learn
```

```python
from core.credit_sentinel.agents.ratio_calculator import RatioCalculator
from core.credit_sentinel.models.distress_classifier import DistressClassifier

ratios = RatioCalculator().calculate_all({
    "total_debt": 500, "ebitda": 100, "interest_expense": 40,
    "total_equity": 200, "current_assets": 150, "current_liabilities": 100,
})
print(ratios)                                   # interest_coverage, leverage, debt_to_equity, current_ratio
print(DistressClassifier().predict_distress(ratios))
```

`core/credit_sentinel/covenant_agent.py` still imports the legacy `adam.core.engine.base.TemplateAgentV26`
and does not import in the current tree; new covenant logic should subclass `TemplateAgentV30`
([AGENTS.md §5](../AGENTS.md#5-standard-interface)). Enterprise memo pipeline demo:
`uv run python scripts/run_credit_memo_pipeline.py`.

## Slice D — System 1 swarm (NeuralMesh)

```bash
uv sync --extra data-science --extra orchestration --extra async-web
python -m core.v30_architecture.python_intelligence.agents.swarm_runner          # agents, port 8001
python -m core.v30_architecture.python_intelligence.bridge.neural_link            # WebSocket bridge, port 8000
```

Contract and ports: [`core/v30_architecture/README.md`](../core/v30_architecture/README.md).
Swarm output is telemetry (`NeuralPacket`); binding decisions go through System 2.

## Slice E — Rust slices

| Crate | Path | Build / test | Python binding |
|---|---|---|---|
| Memory admission | `adam-core/` | `cargo test --manifest-path adam-core/Cargo.toml` | — |
| Risk / stress / PDIL gate | `rust_ext/` | `cargo test --manifest-path rust_ext/Cargo.toml` | PyO3 (abi3-py311): `cd rust_ext && maturin develop --features extension-module` |
| Pricing + matching | `core/rust_pricing/` | `cargo test --manifest-path core/rust_pricing/Cargo.toml` | PyO3: `cd core/rust_pricing && maturin develop` |
| v30 core (ledger, matching, risk) | `core/v30_architecture/rust_core/` | `cargo test --manifest-path core/v30_architecture/rust_core/Cargo.toml` | — |
| AdamOS kernel (lab) | `core/experimental/adamos_kernel/` | `./build.sh` in that directory | — |

Install Rust with `curl https://sh.rustup.rs -sSf | sh` and maturin with `uv pip install maturin`.
Rust kernels are optional: Python callers fall back when the extension is not built. Never hand-edit
`target/` or built `.so` files (Invariant H).

## Slice F — Frontend

| UI | Path | Run |
|---|---|---|
| React client (Vite + TS) | `services/webapp/client/` | `cd services/webapp/client && pnpm install && pnpm dev` (or `npm install && npm run dev`) |
| Flask API for the client | `services/webapp/api.py` (`create_app`) | `uv sync --extra async-web` then `python services/webapp/api.py` (port 5001) |
| Streamlit valuation app | `app.py` | `uv pip install streamlit seaborn matplotlib && streamlit run app.py` |
| Streamlit frontend | `frontend/streamlit_app.py` | `streamlit run frontend/streamlit_app.py` |
| Legacy repo UI (static + live modes) | `run_ui.sh`, `services/ui_backend.py` | `./run_ui.sh` |

The client lockfile is known to drift on `main` (CI baseline); `pnpm run security:regen-lockfile`
regenerates it. Presentation code never goes into `core/` (Path A rule in [AGENTS.md §1](../AGENTS.md#1-bifurcation-core-vs-lab)).

## Slice G — Docker / full stack

```bash
cp .env.example .env
docker compose up -d --build
```

| Service (`docker-compose.yml`) | Port |
|---|---|
| `api` (Flask) | 5001 |
| `core-engine` | 50051 |
| `client` (nginx) | 80 |
| `v24-dashboard` | 3000 |
| `db` (Postgres) / `timescaledb` | 5432 / 5433 |
| `qdrant` | 6333 |
| `neo4j` | 7474 / 7687 |
| `redis`, `worker` | internal |

Layered stacks: `docker compose -f docker-compose.yml -f docker-compose.agents.yml -f docker-compose.infra.yml up -d`.

Individual images: `Dockerfile` (webapp, `docker run -p 5001:5001`), `Dockerfile.core`, `Dockerfile.kernel`,
`Dockerfile.swarm`, `Dockerfile.modern` (all-in-one; add `--gpus all` for GPU hosts).

## Slice H — Custom bundles

```bash
python3 scripts/build_adam.py     # interactive: Lite / Standard / Full profile + module selection
```

Generates a self-contained directory with `requirements.txt` and a `Dockerfile` for the chosen profile.
Details: [custom_builds.md](custom_builds.md).

## Slice I — MCP server

```bash
uv sync --extra data-science --extra orchestration
python server/server.py           # tools are declared in mcp.json
```

## Cloud & Kubernetes

Manifests live in `kubernetes/` (`adam-deployment.yaml`, `adam-os-kernel.yaml`, `sidecar-agents.yaml`,
`adam-secrets.yaml` — replace placeholder secrets before applying). Cloud notes (AWS/GCP/Azure container
services) are in [deployment.md](deployment.md).

## Verify your install

```bash
.venv/bin/python tests/evals/run_accuracy.py                        # routing + entity accuracy
.venv/bin/python -m pytest tests/unit -q                            # module tests (some need optional extras)
.venv/bin/python -m pytest tests/unit/test_standalone_distribution.py -q   # showcase wiring
.venv/bin/ruff check <changed files>
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync` resolution error mentioning opentelemetry | Use extras instead of `requirements/base.txt`; install missing imports individually |
| Test collection errors (`torch`, `flask`, `fpdf`) | Optional deps; install the extra or run a narrower path (`tests/unit/<file>`) |
| `SR11_7_BLOCKED_MODEL_ERROR` | The configured model is in `BLOCKED_MODELS`; change `config/llm_routing.yaml` |
| `GEMINI_API_KEY not found` | Expected without keys; set it in `.env` for live reasoning |
| `{"status": "clarification_needed"}` | Include a ticker or entity name in the query |
| Rust extension not found | Build with `maturin develop` (Slice E) or ignore — Python fallback is used |
| `ModuleNotFoundError: adam.core` | Legacy import path (e.g. `covenant_agent.py`); use `core.` imports |

---

## Historical context

*Adapted from the v23 user/UI manuals (`docs/archive/v23_manual/`) and `docs/archive/SETUP_GUIDE_v23.md`;
version numbers there describe those releases, not the current runtime.*

- **Interaction modes.** The v23 manual introduced intent-based routing: planning ("analyze", "plan"),
  red-team ("attack", "stress test"), ESG ("ESG", "sustainability") and compliance ("KYC", "AML",
  "Basel III"). These survive as `MetaOrchestrator` routes matched by `ROUTE_PATTERNS` (`HIGH`,
  `RED_TEAM`, `ESG`, `COMPLIANCE`; "stress test" now routes to `CRISIS`). Current table:
  [getting_started.md § Interaction modes](getting_started.md#interaction-modes).
- **Critique loop.** Outputs may show a self-critique/revise step — the reflector graph
  (`core/engine/reflector_graph.py`). This is normal.
- **Static vs. live UI.** Static pages work from the filesystem; live features need a backend
  (`run_ui.sh` / `services/ui_backend.py`, or the Flask API). The same split applies to today's showcase.
- **v23 setup** used `pip install -r requirements.txt` and `python server/server.py` for MCP; the MCP
  command is unchanged, the pip path is superseded by uv extras above.
