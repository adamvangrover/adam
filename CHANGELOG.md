# Changelog

## v23.5 - Autonomous Remediation & Enhancement

### Architecture
- **Dependency Management**: Installed critical missing packages (`fastapi`, `flask`, `celery`, `statsmodels`, `semantic-kernel`, `pandera`, `neo4j`, `flask-socketio`, `flask-jwt-extended`, `flask-sqlalchemy`, `flask-cors`, `tweepy`, `pycoingecko`, `feedparser`).
- **Core Schemas**: Updated `HNASP` schema to support `ExecutionTrace` list and `Optional` fields correctly.
- **Base Agent**: Refactored `AgentBase` to improve type safety, fix `jsonLogic` signature, and handle optional `fundamental_epa`.
- **Async Migration**: Refactored `NewsBot` to use `httpx` and `asyncio` for non-blocking I/O.

### Fixes
- **Syntax**: Fixed invalid escape sequences in `core/financial_suite/modules/reporting/generator.py` and `core/risk_engine/engine.py`.
- **Tests**:
    - Created missing `core/v23_graph_engine/data_pipeline/graph.py` to fix `test_adk_data_pipeline.py`.
    - Fixed `unittest.mock` recursion error in `tests/test_interaction_loop.py`.
    - Updated `tests/test_data_retrieval_agent.py` to be async-aware and fixed dependency injection.
    - Updated `tests/test_cyclical_agents.py` to match `ReflectorAgent`'s new output structure.
    - Skipped tests requiring `torch` if not available.
- **Type Safety**: Addressed critical `mypy` errors in core components.
- **Resilience**: Added timeouts to `CatalystAgent` requests. Improved error handling in `NewsBot` for optional ML dependencies.

### Security
- Addressed `bandit` warnings regarding requests without timeout.
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
