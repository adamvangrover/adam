# Autonomous Agent Guide & Drift Documentation

## Overview
This document serves as a guide for autonomous agents working on the Adam codebase. It documents known issues, "drift" (deviations between architecture and implementation), and workarounds.

## 1. Environment & Dependencies
The project uses a complex mix of dependencies. `requirements.txt` is large and can be slow to install. 
- **Drift**: `pip install -e .` does not strictly enforce all dependencies in the test environment if run inside `pipx` (like `pytest`).
- **Guidance**: Always ensure you are running tests using the global python interpreter if you installed dependencies globally. Use `python3 -m pytest` instead of `pytest`.
- **Key Missing Libs**: `pydantic`, `pyyaml`, `pandas`, `numpy`, `torch` (CPU), `fastapi`, `flask`, `redis`.

## 2. InteractionLoop & Async Architecture
The system is transitioning to an asynchronous architecture (v22/v23), but legacy components remain synchronous.
- **Drift**: `InteractionLoop` (`core/system/interaction_loop.py`) is designed as a synchronous loop but orchestrates agents that inherit from `AgentBase`. `AgentBase` enforces an asynchronous `execute` wrapper.
- **Impact**: Synchronous calls to `agent.execute()` in `InteractionLoop` return coroutines, which are not awaited. This breaks the loop's logic (e.g., token counting on coroutine string representation).
- **Workaround**: Tests mock the `execute` method to return values directly, masking the runtime failure. Future work must refactor `InteractionLoop` to be async.

## 3. Mocking & Test Isolation
The test suite heavily relies on mocks, sometimes aggressively.
- **Issue**: "Everything is a Mock". Some tests patch classes or modules globally (or fail to clean up), causing subsequent tests to see Mocks instead of real classes. This leads to `TypeError: object MagicMock can't be used in 'await' expression` or `InvalidSpecError`.
- **Guidance**: 
    - Use `unittest.IsolatedAsyncioTestCase` for async tests.
    - Be careful with `patch.dict(sys.modules)`.
    - If a test fails with "is a Mock", try reloading the module in `setUp` (use `importlib.reload`).

## 4. Configuration Loading
- **Drift**: Agents like `ResultAggregationAgent` or `DataRetrievalAgent` may be initialized without configuration in tests. `load_config` logic inside `__init__` can be fragile if files are missing.
- **Fix**: Code has been patched to handle missing config gracefully (fallback to empty dict), but correct mocking of `load_config` in tests is preferred.

## 5. Echo System
- `Echo` class requires a `config` argument, but was instantiated without it in `InteractionLoop`. This has been patched.

## 6. Frontend
- The frontend is a React app. Use `npm install --legacy-peer-deps` to avoid dependency conflicts.
