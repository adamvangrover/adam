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
