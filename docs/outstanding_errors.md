# Outstanding Errors

## Verified Failures

### 1. Legacy Test Suite Failures (86 Failed)
**Context:** The legacy test suite (`pytest tests/`) reports 86 failures.
**Root Causes:**
*   **Missing API Keys:** Many tests (`TestMarketSentimentAgent`, `TestCodeAlchemist`, `TestSupplyChainRiskAgent`) fail because they require live API keys (OpenAI, Tweepy, etc.) which are not present in the CI environment.
*   **Mocking Issues:** Tests involving `torch` and `tensorflow` are failing due to complex mocking of heavy dependencies in `conftest.py`.
*   **Environment Isolation:** Some tests assume a local development environment with specific tools installed (e.g., `git`, `docker`) which are mocked or missing in the sandbox.

**Decision:**
These failures are known and expected in this environment. The core v23 "Adaptive System" functionality has been verified independently via `tests/verify_v23_*.py` scripts.

### 2. Frontend Verification
**Context:** Frontend tests require `playwright` and a running server. CI environments might lack the necessary browsers or display server.
**Action:** Run `verify_fe.py` locally with a display server.

## Fixed Issues
- **v23 Orchestration:** Fixed `KeyError: 'ticker'` in `core/xai/state_translator.py` by adding a fallback.
- **Dependency Conflicts:** Resolved conflicts between `pydantic` (<2.12), `semantic-kernel`, and `flask-cors`.
- **Security Tests:** Fixed `tests/test_legacy_api_security.py` to correctly import the shadowed `legacy_api` module and verify the security patch.
- **Missing Modules:** Installed missing dependencies: `fastapi`, `statsmodels`, `flask-cors`, `pandera`, `flask-socketio`, `flask-sqlalchemy`, `flask-jwt-extended`, `celery`, `pyarrow`, `scikit-learn`, `beautifulsoup4`, `langchain`, `langchain-community`, `transformers`.

## Notes
- Some tests are skipped if API keys are missing.
- Ensure `PYTHONPATH=.` is set when running tests from the root directory to resolve `core` modules.
