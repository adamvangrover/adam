# v30 Architecture — forward path

`core/v30_architecture/` is the forward runtime path (not a snapshot). It is **System 1**
(AGENTS.md §2): async agents that publish packets to a NeuralMesh, plus a Rust core.

| Slice | Path | Run |
|---|---|---|
| NeuralMesh agents | `python_intelligence/agents/` | `python -m core.v30_architecture.python_intelligence.agents.swarm_runner` (port 8001) |
| Mesh bridge (WebSocket pub/sub) | `python_intelligence/bridge/neural_mesh.py`, `neural_link.py` | `neural_link.py` serves on port 8000 |
| Intent orchestrator | `python_intelligence/orchestrator/v30_orchestrator.py` (`EACIOrchestrator`) | `process_intent_async(intent, user_role, context)` |
| Workflow engine | `python_intelligence/workflows/sovereign_workflow.py` | `SovereignWorkflowEngine.execute_task(task)` |
| MCP tool server (HITL-gated sensitive tools) | `python_intelligence/mcp/server.py` | `MCPServer.call_tool(name, args)` |
| Rust core | `rust_core/` | `cargo test --manifest-path core/v30_architecture/rust_core/Cargo.toml` |
| Neuro-quantum experiments | `neuro_quantum/` | `run_assessment.py` |

## Contract (deliberately different from System 2)

System 2 agents (`core/agents/`) use `AgentInput` → `AgentOutput` from `core/schemas/agent_schema.py`.
v30 `BaseAgent` does **not**: it is a long-running async actor.

- Identity: `BaseAgent(name, role)`.
- Lifecycle: `async run()` loops forever; optional `async execute(...)` performs one unit of work.
- Output: `await self.emit(packet_type, payload)` → `NeuralPacket` (`id`, `timestamp`, `source_agent`,
  `packet_type`, `payload`, `priority`) on the mesh. Consumers subscribe by topic; nothing is returned.
- Typed I/O is per-agent Pydantic (e.g. `FundamentalInput`/`FundamentalOutput`), not a shared schema.

To hand a v30 result to System 2, put it in `AgentInput.context`; to publish a System 2 result to the
mesh, emit `AgentOutput.model_dump()` as a packet payload. No adapter class exists on purpose.

## Audit (vs. the Phase-2 contract)

| Component | Finding |
|---|---|
| `agents/*` `execute` signatures | Inconsistent: `execute(**kwargs)` (macro, compliance, red team, shield, sovereign, news) vs. `execute(input_data: FundamentalInput)`. Some agents only have `run()`. |
| Provenance | Packets carry no `ProvenanceHeader`; v30 output is telemetry, not a material decision. Decisions must go through System 2 to get `provenance_trace`. |
| `orchestrator/v30_orchestrator.py` | Has a `MockMCPServer` fallback and a `MockCreditAgent` demo; keyword plan decomposition. |
| `mcp/server.py` | `execute_trade` / `get_market_data` are demo tools; sensitive tools require HITL approval. |
| Tests | `tests/test_*v30*`, `test_neural_mesh_advanced.py`, `test_sovereign_swarm.py`, `test_macro_economic_agent.py`, `test_news_bot_v30.py`. |
