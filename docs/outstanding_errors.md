# Outstanding Errors

## Verified Failures

### 1. Test Suite Isolation
**Context:** Many tests pass individually but fail when run in the full suite (e.g. `pytest tests/`). This is due to global state pollution (mocking `sys.modules`) in `conftest.py` and individual tests not cleaning up.
**Action:** Run tests individually or refactor tests to use `patch.dict(sys.modules, ...)` instead of modifying `sys.modules` directly.

### 2. Frontend Verification
**Context:** Frontend tests require `playwright` and a running server. CI environments might lack the necessary browsers or display server.
**Action:** Run `verify_fe.py` locally with a display server.

## Fixed Issues
- `tests/api/test_service_state.py`: Fixed 500 Error by patching `torch` and `tensorflow` mocks in `conftest.py` to support `importlib` and `torch._dynamo` inspection.
- `TypeError` in `TestClient`: Fixed by pinning `httpx<0.28.0`.
- Security: Fixed hardcoded `debug=True`.

## Notes
- Some tests are skipped if API keys are missing.
- Ensure `PYTHONPATH=.` is set when running tests from the root directory to resolve `core` modules.
