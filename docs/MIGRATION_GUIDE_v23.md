# Adam v23.0 Migration Guide

This document outlines the key changes, robustness enhancements, and migration steps for the Adam v23.0 "Adaptive System".

## 1. Deprecation & Modernization

### 1.1 `legacy_api`
The `core/api.py` module (often referred to as `legacy_api`) is now marked as deprecated.
New endpoints should be implemented in `services/webapp/api.py`.

Use the `@deprecated` decorator from `core.utils.deprecation` when maintaining legacy code:

```python
from core.utils.deprecation import deprecated

@deprecated(version="23.0", replacement="services.webapp.api")
def old_function():
    pass
```

### 1.2 Test Suite Hygiene ("Toxic Tests")
Some legacy integration tests modify global state (e.g., `sys.modules`) which pollutes the environment for subsequent tests.
**Policy:** Tests that perform invasive mocking of global modules must be renamed to start with `z_` (e.g., `tests/z_test_api_v23_wiring.py`) to ensure they run last in the test suite.

## 2. Robustness Enhancements

### 2.1 Agent Orchestrator
The `AgentOrchestrator` now supports:
- **`health_check()`**: Returns the status of all loaded agents.
- **`execute_agent_safe()`**: Automatically falls back to a generic agent (e.g., `GeneralAgent`) if the specific agent fails.

### 2.2 Risk Assessment Agent
- **Caching**: Risk assessments are now cached in-memory (LRU, 5 min TTL) to reduce redundant computation.
- **Graph Fallback**: Automatically falls back to v21 logic if the v23 `CyclicalReasoningGraph` cannot be loaded.

### 2.3 Universal Ingestor
- **Safe Mode**: Automatically skips files larger than 10MB to prevent MemoryErrors.
- **Enhanced Support**: Now processes `.log` files for metadata.

## 3. Migration Steps

1. **Update Imports**: Replace direct imports from `core.api` with `services.webapp.api` where applicable.
2. **Enable Caching**: Verify `RiskAssessmentAgent` configuration if persistence is needed (currently in-memory only).
3. **Verify Tests**: Run `pytest` and ensure no `ImportError` or `ModuleNotFoundError` occur. Use `z_` prefix for tests that mock `sys.modules`.

## 4. Troubleshooting

*   **Error:** `TypeError: object MagicMock can't be used in 'await' expression`
    *   **Cause:** A test mocked a dependency (like `semantic_kernel`) globally and didn't clean it up.
    *   **Fix:** Ensure the polluting test has a robust `tearDown` or runs last (`z_` prefix).

*   **Error:** `ModuleNotFoundError: No module named 'legacy_api'`
    *   **Cause:** Shadowing of `core/api.py` by `core/api/` package.
    *   **Fix:** Use `importlib` to load the file explicitly by path.
