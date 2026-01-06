# Outstanding Errors & Fixes

## Recently Fixed
- **Missing Dependencies:** Installed `pytest`, `pydantic`, `numpy`, `pyyaml`, `python-json-logger`, `langgraph`, `networkx`, `tiktoken`, `pandas`, `semantic-kernel`, `textblob`, `tweepy`, `scikit-learn`, `beautifulsoup4`, `langchain`, `langchain-community`.
- **Broken Imports:**
    - Fixed `core/schemas/__init__.py` to remove imports of non-existent modules (`hnasp_integration`, etc.).
    - Fixed `core/engine/meta_orchestrator.py` to correctly import `SNCRatingAgent` and `CovenantAnalystAgent` from their actual locations in `core/agents/specialized/`.
- **Test Failures:**
    - `tests/test_agent_orchestrator.py`: NOW PASSING.
    - `tests/test_v23_5_pipeline.py`: NOW PASSING.
- **CI/CD:**
    - Created `.github/workflows/ci.yml` for automated testing and linting across Python 3.10-3.12.
- **Features:**
    - Implemented `MemoryMixin` for agent state persistence.
    - Reorganized `prompt_library` into `AOPL-v1.0` standard.

## Remaining Known Issues

### 1. Legacy Test Suite Failures
**Context:** The full legacy test suite (`pytest tests/`) may still report failures due to environment isolation (missing API keys for `TestMarketSentimentAgent`, etc.).
**Action:** Focus on `verify_v23_*.py` scripts and specific unit tests for active development.

### 2. Frontend Verification
**Context:** Frontend tests require `playwright` and a running server.
**Action:** Run `verify_fe.py` locally if modifying UI components.

## Notes
- Always run tests with `PYTHONPATH=.` from the root directory.
- `requirements.txt` may need further pruning of unused dependencies (e.g., conflicting `flask` versions if any).
