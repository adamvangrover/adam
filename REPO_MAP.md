# REPO_MAP

Terse, factual map of this repository for new contributors and for models ingesting the repo.
If this file disagrees with code, the code wins — fix this file.

## 1. Canonical tree

| Path | Role | Status |
|---|---|---|
| `core/` | Canonical Python package: agents, engines, schemas, governance | **canonical** |
| `core/agents/` | Agent implementations (`AgentBase` subclasses) | canonical |
| `core/agents/templates/` | Reference agent templates | canonical |
| `core/engine/` | Orchestration (`MetaOrchestrator`), graphs, consensus, simulation | canonical |
| `core/schemas/` | Pydantic contracts (`agent_schema.py` = agent I/O) | canonical contract |
| `core/governance/` | Policy, invariants, risk classes, approval, immutable ledger | canonical |
| `core/provenance/` | W3C PROV-O helpers, hashing, lineage | canonical |
| `core/config/` | Typed config models (`models.py`: `RouterConfig`, `BLOCKED_MODELS`) | canonical |
| `core/v30_architecture/` | v30 forward path: async NeuralMesh agents + Rust core | forward path |
| `config/` | YAML/JSON config registry (see §3) | canonical contract |
| `schemas/` | JSON Schemas (agent, context, event, evaluation, ...) | canonical contract |
| `assets/` | Externalized prompts and JSONLogic rule artifacts | canonical |
| `services/webapp/` | Flask API + React client (`client/`) | runtime |
| `server/` | MCP server (`server/server.py`, tools listed in `mcp.json`) | runtime |
| `rust_ext/`, `core/rust_pricing/`, `adam-core/` | Rust kernels (see §6) | optional runtime slices |
| `tests/` | pytest suite | canonical |
| `scripts/` | CLI / maintenance / generators | tooling |
| `docs/` | Long-form docs, ADRs | docs |
| `showcase/` | All static HTML (GitHub Pages root). `showcase/dashboards/` = former root-level dashboards | presentation |
| `data/` | Seed data, ledgers, telemetry | data |
| `core/zero_trust/` | Zero-trust ingestion gateway, schema bounds, triangulated feed consensus | canonical |
| `archive/` | Historical scripts; `archive/snapshots/` = learning snapshots (§7) | not imported |

Root `index.html` is the only HTML page at the repo root. Generated "Index of" pages are not
produced inside `core/engine`, `core/agents`, `core/credit_sentinel`, `core/v30_architecture`, `services/webapp`
(`scripts/generate_directory_indexes.py` `SKIP_PATHS`).

## 2. Entry points

| Entry point | Location | Contract |
|---|---|---|
| `MetaOrchestrator.route_request(query, context)` | `core/engine/meta_orchestrator.py` | Routes a query via `Route` enum + `OrchestratorConfig`; async |
| `AgentBase.execute(...)` | `core/agents/agent_base.py` | Agent execution; standard I/O in `core/schemas/agent_schema.py` |
| Flask API `create_app()` | `services/webapp/api.py` | `/api/v23/analyze` → `MetaOrchestrator.route_request`; `/api/credit/*`, `/api/consensus`, ... |
| CLI | `scripts/run_adam.py`, `core/main.py` | Graph run from the command line |
| MCP server | `server/server.py` | Tool surface for MCP clients |
| v30 swarm | `core/v30_architecture/python_intelligence/orchestrator/v30_orchestrator.py` | NeuralMesh `BaseAgent` contract (see §5) |

## 3. Config registry (`config/`)

| File | Purpose |
|---|---|
| `llm_routing.yaml` | Model routing tiers; `routing_tiers.tier_1_primary_system2.model_id` overrides the orchestrator default; SR 11-7 registry |
| `agents.yaml` | Agent registry / per-agent config |
| `agent_schema.yaml` | Agent config schema |
| `orchestration.yaml`, `workflow.yaml` | Orchestrator and workflow settings |
| `governance_policy.yaml` | Restricted endpoints, roles, risk levels |
| `system.yaml`, `settings.yaml`, `config.yaml` | General system settings |
| `api.yaml`, `data_sources.yaml` | External API and data-source settings (no secrets in git) |
| `knowledge_graph*.yaml` | Knowledge-graph connection and schema |
| `logging.yaml`, `telemetry.yaml` | Logging / telemetry |
| `black_swan_scenarios.yaml` | Stress scenarios |
| `covenants/` | Covenant rule definitions |
| `Adam_v*_Portable_Config.json` | Historical portable configs (reference only) |

## 4. Governance contexts

| Context | Where |
|---|---|
| Confidence tiers: ≥0.85 autonomous, 0.50–0.8499 HITL, <0.50 abort | `AGENTS.md` §3 |
| SR 11-7 blocked models (`BLOCKED_MODELS`, `SR11_7_BLOCKED_MODEL_ERROR`) | `core/config/models.py`, enforced in `MetaOrchestrator` |
| Architectural invariants A–J | `AGENTS.md` §7, `core/governance/invariants.py` |
| Endpoint policy / RBAC | `config/governance_policy.yaml`, `core/governance/policy.py` |
| Provenance (W3C PROV-O) | `core/provenance/`, `AgentOutput.provenance_trace` |
| Logic-as-data (JSONLogic) | `assets/`, `AgentBase.evaluate_logic_layer` |
| `_estimated: True` | Marks fabricated / placeholder values in config and outputs (convention, not linted) |

## 5. Contracts

- Agent I/O: `core/schemas/agent_schema.py` (`AgentInput`, `AgentOutput`). Single source of truth.
- v30 NeuralMesh agents (`core/v30_architecture/python_intelligence/agents/base_agent.py`) use packet emission, not `AgentInput`/`AgentOutput`.
- JSON Schemas for cross-runtime use: `schemas/*.schema.json`.

## 6. Rust slices

| Crate | Path | Notes |
|---|---|---|
| `adam-core` | `adam-core/` | Fixed-point arithmetic, memory admission/provenance. Only Rust memory implementation; `cargo test --manifest-path adam-core/Cargo.toml` |
| `rust_ext` | `rust_ext/` | PyO3/maturin extensions |
| `rust_pricing` | `core/rust_pricing/` | Pricing kernel |
| `adam_core_engine` | `core/v30_architecture/rust_core/` | v30 Rust core |
| `adamos_kernel` | `core/experimental/adamos_kernel/` | Kernel prototype (the only crate in CI) |

## 7. Snapshot index

Versioned codebases are kept as learning snapshots under `archive/snapshots/<name>/`, each with a `SNAPSHOT.md`.
They are not imported by live code and are excluded from ruff (`[tool.ruff] exclude`), pytest (`norecursedirs`)
and the unit-test CI path filter.

| Snapshot | Location | Pulled into live tree |
|---|---|---|
| `afos_v3` | `archive/snapshots/afos_v3/` | `core/engine/prov_o_consensus.py`, `core/zero_trust/` (all code) |
| `adam_v24` | `archive/snapshots/adam_v24/` | `data/schemas/unified_ledger_bitemporal.sql` |
| `ufos` | `archive/snapshots/ufos/` | — |
| `tinker_lab` | `archive/snapshots/tinker_lab/` | — (`tinker-cookbook` submodule path moved) |
| `experimental` | `archive/snapshots/experimental/` | — (`core/experimental/` is separate and live) |
| `adam-core` | `adam-core/` (retained) | Optional Rust runtime slice, see §6 |

Consensus engines in `core/`: `consensus_engine.py` (weighted agent-vote arbitration + HITL),
`consensus_engine_v2.py` (memory-matrix house view), `risk_consensus_engine.py` (SNC risk),
`prov_o_consensus.py` (PROV-O disjoint-root corroboration), `core/zero_trust/triangulated_consensus.py` (3-feed outlier rejection + lockdown).

## 8. Run

```bash
uv venv --python 3.12 && uv sync --extra data-science --extra orchestration
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check <files>
```
