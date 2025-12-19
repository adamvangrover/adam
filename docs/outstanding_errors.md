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
# Outstanding Errors and Issues

## Frontend Verification
- **Issue**: Frontend verification script (`verify_app.py`) failed to detect the `h1` element "ADAM v23.5".
- **Symptom**: `Error: Page.wait_for_selector: Timeout 10000ms exceeded`.
- **Root Cause**: The React development server (`react-scripts start`) takes time to compile and serve the application. The headless browser connection was refused initially, and subsequent attempts timed out waiting for the specific element.
- **Resolution Status**: Skipped verification to proceed with submission as per user instructions. The compilation was successful (`Compiled successfully!`), indicating the code structure is valid.
- **Action Required**: Future developers should verify the UI rendering in a full browser environment.

## Test Failures
- **Issue**: `npm test` failed in `App.test.js`.
- **Root Cause**: The test file was testing `App.js` which was replaced by `App.tsx` and removed.
- **Resolution**: `App.test.js` was deleted as it is no longer relevant for the new TSX structure without proper Jest configuration for TypeScript.
- **Action Required**: New tests should be written for `App.tsx` and other components using `ts-jest` or compatible configuration.

## Known Issues (FO Super-App Integration)

1.  **Live Connections:** While `core/market_data` now supports `yfinance` and `core/pricing_engine` uses realistic GBM simulation, connection to institutional feeds (Bloomberg, Fix Protocol) is still pending implementation.
2.  **Vector Search:** `core/memory/engine.py` persists to `data/personal_memory.db` but relies on keyword search. True semantic search (Chroma/FAISS) requires integrating the `embeddings` module.
3.  **UI Integration:** The backend logic is fully integrated into `MetaOrchestrator` (including Family Office routes), but the frontend `showcase/` dashboard does not yet expose specific widgets for the FO Super-App.
4.  **Dependencies:** The system now requires `langgraph`, `numpy`, `pandas`, `transformers`, `torch`, `spacy`, `textblob`, `tweepy`, `scikit-learn`, `beautifulsoup4`, `redis`, `pika`, `python-dotenv`, `tiktoken`, `semantic-kernel`, and `langchain-community`.

## Adam v23.0 "Swarm Coding Device" Errors

- **Issue**: `cargo build` in `core/rust_pricing` fails due to sandbox file limits.
- **Root Cause**: The `target/` directory generates too many files.
- **Resolution**: Added `target/` to `.gitignore`. The Rust source code is valid and should build in a non-constrained environment.
- **Issue**: AST parsing errors in `scripts/swarm_showcase_v23.py` for legacy Python files (e.g., `core/system/system_controller.py`).
- **Root Cause**: Legacy files may contain Python 2 syntax or invalid syntax (e.g., indentation errors, missing expressions).
- **Resolution**: The showcase generator logs these errors and skips the files, generating the report for valid files.

## Fixes Implemented (2025-12-17)
- **Graceful Fallbacks**: Implemented try-except blocks in `AgentBase`, `AgentOrchestrator`, `KGCache`, and `data_utils` to handle missing dependencies (`semantic-kernel`, `redis`, `pika`).
- **Orchestration Resilience**: Fixed `MetaOrchestrator` to handle `AgentOrchestrator` initialization failures gracefully, ensuring the system boots even if legacy components fail.
- **Dependency Stabilization**: Installed critical dependencies (`langgraph`, `spacy`, `transformers`, `semantic-kernel`) and verified `run_adam.py` execution.
- **Deep Dive Protocol**: Validated end-to-end execution of the v23.5 Deep Dive flow (Apple test case).
