# Outstanding Errors and Technical Debt

This document lists known errors and technical debt that should be addressed in future development cycles.

## 1. Fragile Dependency Management

**Issue:** The project's dependencies are not well-managed, leading to frequent installation failures and conflicts. The `requirements.txt` file is not always compatible with the latest versions of Python or other packages.

**Example:**
- `statsmodels==0.13.2` fails to build with recent Python versions. It was temporarily upgraded to `0.14.2` to allow installation, but a full dependency review is needed.
- Installing `facebook-scraper` causes version conflicts with `pyee` and `websockets`, breaking other packages like `playwright` and `semantic-kernel`.
- The `lxml` library has moved its `html.clean` module to a separate project (`lxml_html_clean`), which is not reflected in the dependencies, causing `ImportError` when `facebook-scraper` is used.

**Recommendation:**
- Perform a full audit of all dependencies.
- Pin all top-level dependencies to known working versions.
- Consider using a more robust dependency management tool like Poetry or Pipenv to manage transitive dependencies and resolve conflicts.
- Create a `requirements-dev.txt` for testing and linting packages.

## 2. Incomplete Test Suite

**Issue:** Many tests fail to run due to `ModuleNotFoundError` or other import errors, indicating that the test environment is not properly configured or that the tests have not been maintained.

**Recommendation:**
- Fix all dependency issues to allow the full test suite to be collected.
- Run the full suite and triage all failures.
- Implement a CI/CD pipeline that runs the tests on every commit to prevent future regressions.

# Outstanding Errors

This document lists known errors and issues in the codebase that need to be addressed in future iterations.

## Test Failures

The following tests are currently failing:

*   `scripts/test_new_agents.py`
*   `tests/test_agent_orchestrator.py`
*   `tests/test_agents.py`
*   `tests/test_interaction_loop.py`
*   `tests/test_system.py`
*   `tests/test_v21_orchestrator_loading.py`

The root cause of these failures is a `ModuleNotFoundError` for `facebook_scraper`.

## Dependency Conflicts

There is a dependency conflict between `facebook-scraper` and `semantic-kernel`.

*   `facebook-scraper` requires `pyppeteer`, which in turn requires `pyee<12.0.0` and `websockets<11.0`.
*   `semantic-kernel` requires `pyee>=13.0.0` and `websockets>=13.0`.

This conflict prevents `facebook-scraper` from being installed alongside `semantic-kernel`. The current workaround is to not install `facebook-scraper`, which causes the tests listed above to fail. This needs to be resolved in a future iteration.
