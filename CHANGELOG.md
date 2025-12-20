# Changelog

## [Unreleased] - 2025-05-20 (Simulated)

### Fixed
- **Critical:** Resolved `ModuleNotFoundError` for `core.v23_graph_engine.data_pipeline.graph` by implementing the missing graph definition.
- **Critical:** Fixed `tests/test_interaction_loop.py` mocking logic to support `AsyncMock` and correct `AgentOrchestrator` patching.
- **Core:** Refactored `core/agents/agent_base.py` to safely handle `asyncio` loops in threaded contexts and robustify `update_persona` against None values.
- **Core:** Refactored `core/system/interaction_loop.py` to correctly inject `config` into `Echo` and `check_token_limit`, and wrap async agent calls with `asyncio.run`.
- **Core:** Fixed `core/agents/query_understanding_agent.py` to call synchronous `LLMPlugin.generate_text` instead of non-existent `get_completion`.
- **Ops:** Installed missing dependencies: `pydantic`, `flask`, `torch` (CPU), `textblob`, `langchain-community`, `json-logic`.

### Added
- `reproduce_api_error.py` script to debug API endpoint failures.
- `RemediationPlan.json` outlining future steps for 100% system integrity.

### Known Issues
- `tests/api/test_service_state.py` fails with 500 error on Adam-mini endpoint (likely optimizer logic).
- Multiple agent tests failing due to strict mocking or logic divergence in v23 transition.
