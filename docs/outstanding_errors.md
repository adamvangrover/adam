# Outstanding Errors and Technical Debt

This document lists known errors and technical debt that should be addressed in future development cycles.

## 1. Fragile Dependency Management

**Issue:** The project's dependencies are not well-managed.

**Update (v23):**
- The conflict between `facebook-scraper` and `semantic-kernel` has been **resolved** by removing `facebook-scraper` from requirements and making it an optional import in `core/data_sources/social_media_api.py`.
- `langgraph`, `tiktoken`, `pandas`, `matplotlib`, `seaborn` have been verified as required dependencies and added to the environment.

**Recommendation:**
- Perform a full audit of all dependencies.
- Pin all top-level dependencies to known working versions.
- Consider using a more robust dependency management tool like Poetry or Pipenv.

## 2. Incomplete Test Suite

**Issue:** Many legacy tests fail to run.

**Update (v23):**
- `tests/verify_v23_full.py` PASSES.
- Legacy tests that depend on `facebook-scraper` will fail or skip gracefully if handled correctly.
- `scripts/run_adam.py` is functional and acts as the new entry point.

**Recommendation:**
- Run the full suite and triage all failures.
- Implement a CI/CD pipeline.

## 3. Environment Limitations (v23)

**Issue:** The current runtime environment lacks `langgraph` and `numpy`, preventing the execution of `scripts/test_sentiment_graph.py`.

**Detail:**
- `core/v23_graph_engine/market_sentiment_graph.py` relies on `langgraph`.
- The `test_sentiment_graph.py` script was run with mocks to verify syntax and logical structure, which passed.
- **Action Required:** Install `langgraph`, `numpy`, and `pandas` in the target environment to fully enable v23 features.
