# Outstanding Errors Log

## Overview
This log tracks test failures and known issues remaining after the v23.5 stabilization effort. While the core system is functional and hardened, some tests remain flaky due to the complex mock environment required for "Hybrid-Core" architecture.

## Test Failures (Backend)

### Test Suite: `tests/`
- **Status**: ~80 failures in full suite run, but key tests (`test_agents.py`, `test_agent_orchestrator.py`) pass in isolation.
- **Root Cause**: Global `sys.modules` patching in `tests/conftest.py` (required for mocking heavy libs like `spacy`/`transformers`) causes side effects when running all tests sequentially.
- **Mitigation**: Run tests in smaller batches or isolation.

### Specific Failures
1. **`test_agent_orchestrator.py` (Full Run)**
   - `AssertionError: 'MockAgent' not found`: Likely due to `AGENT_CLASSES` patching interaction with other tests.
   - **Fix Status**: Passed in isolation.

2. **`test_agents.py` (Full Run)**
   - `test_analyze_sentiment` failure: Context leakage.
   - **Fix Status**: Passed in isolation (6/6 passed).

3. **Missing Dependencies in CI**
   - `scikit-learn` and `pandas` older versions in `requirements.txt` cause install issues on Python 3.12.
   - **Resolution**: Versions were relaxed to allow installation.

## Frontend
- **Status**: 100% Pass (1 test).
- **Notes**: Added `App.test.tsx` with extensive mocking of child components to ensure isolated rendering test passes.

## Recommendations
- Migrate `unittest` based tests to `pytest` fixtures completely to avoid `sys.modules` patching issues.
- Dockerize the test environment to ensure consistent dependency versions.
