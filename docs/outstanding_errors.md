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
