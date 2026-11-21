# Architecture

Short module map. File locations: [`REPO_MAP.md`](../REPO_MAP.md). Agent rules: [`AGENTS.md`](../AGENTS.md).
Overview, diagrams and glossary: [`README.md`](../README.md#-system-architecture). Setup per slice:
[`setup_guide.md`](setup_guide.md). Presentation templates (brands, newsletters, terminals, prompts):
[`TEMPLATES.md`](TEMPLATES.md).

```
request ──► services/webapp/api.py (Flask)  /  scripts/run_adam.py  /  server/server.py (MCP)
                 │
                 ▼
     core/engine/meta_orchestrator.py :: MetaOrchestrator.route_request(query, context)
        1. SR 11-7 model gate (core/config/models.py::BLOCKED_MODELS)
        2. route = force_route | ROUTE_PATTERNS (ordered \b regex) | LOW
        3. handler(query, context)  ── unresolved entity ──► {"status": "clarification_needed"}
                 │
   ┌─────────────┼───────────────────────────────┬──────────────────────────┐
   ▼             ▼                               ▼                          ▼
 LangGraph     Family-office handlers          Agent registry           System 1 swarm
 graphs        (wealth / deal / execution /    (MEDIUM, LOW via         core/v30_architecture/
 core/engine/  market)                         AgentBase.execute)        (NeuralMesh packets)
 *_graph.py
   │
   ▼
 Deterministic engines: crisis/sector/contagion (core/engine/), valuation, Rust kernels
   │
   ▼
 AgentOutput (core/schemas/agent_schema.py) + provenance_trace ──► governance ledger (core/governance/)
```

## Layers

| Layer | Modules | Contract |
|---|---|---|
| Entry | `services/webapp/`, `server/`, `scripts/run_adam.py` | HTTP / MCP / CLI |
| Orchestration | `core/engine/meta_orchestrator.py`, `core/engine/semantic_router.py` | `Route` enum, `OrchestratorConfig` |
| System 2 agents | `core/agents/` (`AgentBase`, `PydanticAgentBase`, `templates/TemplateAgentV30`) | `AgentInput` → `AgentOutput` |
| Graphs | `core/engine/*_graph.py` (deep dive, crisis, red team, ESG, compliance, SNC, reflector) | state in → state diff out |
| Engines | `core/engine/` (consensus, crisis, sector impact, contagion), `core/credit_sentinel/` | deterministic, no LLM math |
| System 1 | `core/v30_architecture/python_intelligence/` | `BaseAgent.emit` → `NeuralPacket` |
| Ingestion | `core/zero_trust/` (gateway, schema bounds, triangulated consensus), `src/ingestion/` | Pydantic bounds; lockdown on disagreement |
| Governance | `core/governance/`, `config/governance_policy.yaml`, `src/pdil/` | risk class + immutable event per state change |
| Provenance | `core/provenance/`, `src/pdil/models.py::ProvenanceHeader` | `docs/PROVENANCE.md` |
| Rust | `adam-core/` (memory), `core/rust_pricing/`, `rust_ext/`, `core/v30_architecture/rust_core/` | `cargo test` |
| Presentation | `showcase/`, `frontend/`, `services/webapp/client/` | reads outputs only |

## Configuration

`config/*.yaml` is the registry (index in `REPO_MAP.md` §3). Key files: `llm_routing.yaml` (model tiers;
tier 1 overrides the orchestrator default), `agent_schema.yaml` (confidence tiers),
`default_universe_metadata.yaml` (crisis position metadata), `governance_policy.yaml`.
Rules: `assets/` JSONLogic.

## Tests

| Dir | Purpose |
|---|---|
| `tests/unit/` | Module tests |
| `tests/regression/` | One seam test per fixed bug |
| `tests/evals/` | JSON datasets + `run_accuracy.py` (routing, entity) with accuracy floors |
| `tests/integration/`, `tests/<domain>/` | Cross-module and domain suites |
