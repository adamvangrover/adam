<img width="2816" height="1536" alt="Gemini_Generated_Image_1abr9w1abr9w1abr" src="https://github.com/user-attachments/assets/2a25b3fd-005d-465c-b51b-ffca4bbf13e5" />

---

<div align="center">
  <a href="https://adamvangrover.github.io/adam/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="Adam OS Logo" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>Autonomous Deterministic Alpha Matrix</h3>
  <p><em>The Institutional-Grade Neuro-Symbolic Financial Sovereign</em></p>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://hub.docker.com/" target="_blank"><img src="https://img.shields.io/badge/docker-ready-blue" alt="Docker Ready"></a>
  <a href="AGENTS.md"><img src="https://img.shields.io/badge/Architecture-v30.1-blueviolet" alt="v30.1"></a>
  <a href="docs/setup_guide.md"><img src="https://img.shields.io/badge/docs-comprehensive-brightgreen" alt="Docs"></a>
</div>

<br>

**Version:** 30.1 &nbsp;|&nbsp; **Domain:** Institutional credit risk & market intelligence

---

ADAM is a multi-agent framework that keeps LLM reasoning bounded by deterministic risk engines,
externalized rules (JSONLogic/YAML) and W3C PROV-O provenance.

**Start here**

| You are | Read |
|---|---|
| New contributor / model ingesting the repo | [`REPO_MAP.md`](REPO_MAP.md) (where everything is), [`machine_index.json`](machine_index.json) |
| Building agents | [`AGENTS.md`](AGENTS.md), [`core/agents/templates/v30_template_agent.py`](core/agents/templates/v30_template_agent.py) |
| Understanding the flow | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Working on System 1 / v30 | [`core/v30_architecture/README.md`](core/v30_architecture/README.md) |
| Auditing outputs | [`docs/PROVENANCE.md`](docs/PROVENANCE.md) |

## What is in the repo

| Path | What |
|---|---|
| `core/` | Canonical Python package: agents, `MetaOrchestrator`, LangGraph graphs, engines, governance, provenance |
| `core/schemas/agent_schema.py` | `AgentInput` / `AgentOutput` — the one agent contract |
| `core/v30_architecture/` | System 1 NeuralMesh swarm + Rust core (forward path) |
| `config/` | YAML config registry (model routing, confidence tiers, governance, universe metadata) |
| `assets/` | JSONLogic rules and prompts |
| `services/webapp/` | Flask API (`api.py::create_app`) + React client |
| `server/` | MCP server (tools in `mcp.json`) |
| `adam-core/`, `core/rust_pricing/`, `rust_ext/` | Optional Rust slices |
| `showcase/` | Static HTML dashboards (GitHub Pages) |
| `archive/snapshots/` | Frozen earlier codebases (`adam_v24`, `afos_v3`, `ufos`, `tinker_lab`, `experimental`), each with `SNAPSHOT.md`; not imported |

## Quick start

Requires Python 3.11+ and [`uv`](https://astral.sh/uv). Rust is optional.

```bash
git clone https://github.com/adamvangrover/adam.git && cd adam
uv venv --python 3.12
uv sync --extra data-science --extra orchestration   # or --extra full-kernel
uv run python scripts/run_adam.py
```

`requirements/base.txt` has conflicting pins; if `uv sync` fails, install the failing imports individually
(`uv pip install <pkg>`).

Without an LLM key the orchestrator falls back to mock behaviour. The default orchestrator model is
`gemini-4-neon`, overridable in `config/llm_routing.yaml`; SR 11-7 blocked models are refused at startup.

Other entry points:

```bash
docker compose up -d                                   # full stack
cargo test --manifest-path adam-core/Cargo.toml        # Rust memory slice
```

Open `showcase/index.html` for the static dashboards (no backend needed).

## Using the orchestrator

```python
import asyncio
from core.engine.meta_orchestrator import MetaOrchestrator

orch = MetaOrchestrator()
print(asyncio.run(orch.route_request("Simulate a market crash", context={"portfolio": {"aum": 1e6, "allocation": {"SPY": 0.6, "TLT": 0.4}}})))
```

Queries route by ordered word-boundary patterns (deep dive, swarm, code gen, red team, crisis,
surveillance, ESG, compliance, family-office wealth/deal/execution/market, HIGH/MEDIUM/LOW). Queries
without a resolvable ticker or entity return `{"status": "clarification_needed"}`.

## Tests and evals

```bash
.venv/bin/python -m pytest tests/ -q          # everything
.venv/bin/python -m pytest tests/unit -q      # module tests
.venv/bin/python -m pytest tests/regression -q
.venv/bin/python tests/evals/run_accuracy.py  # routing + entity accuracy vs. floors
.venv/bin/ruff check <changed files>
```

Some suites need optional dependencies and fail to collect without them.

## Governance

Every state-changing operation gets a risk class and an immutable event (`core/governance/`,
`config/governance_policy.yaml`). Agent confidence: >= 0.85 autonomous, 0.50–0.8499 human approval,
< 0.50 abort (`config/agent_schema.yaml`). Material decisions carry a `provenance_trace`.

## More

[`docs/setup_guide.md`](docs/setup_guide.md) · [`AGENT_CATALOG.md`](AGENT_CATALOG.md) ·
[`docs/AGENTS_KNOWLEDGE_BASE.md`](docs/AGENTS_KNOWLEDGE_BASE.md) · [`SECURITY.md`](SECURITY.md) ·
[`CHANGELOG.md`](CHANGELOG.md)

## License

MIT — see [`LICENSE`](LICENSE).
