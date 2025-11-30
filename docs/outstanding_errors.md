# Outstanding Errors and Technical Debt

This document lists known errors and technical debt that should be addressed in future development cycles.

## 1. Fragile Dependency Management

**Status:** IN PROGRESS
- **Update:** `scripts/run_adam.py` and `tests/verify_v23_full.py` now run successfully in the v23 environment.
- **Action:** `requirements.txt` has been cleaned up.
- **Remaining:** Some legacy v21 agents may still require pinned versions of `tensorflow` or `torch-sparse` which are currently disabled to favor `torch` CPU builds.

## 2. API Key Dependencies

**Status:** MITIGATED
- **Update:** The system now gracefully handles missing API keys for `Cohere`, `OpenAI`, and `Google Search` by initializing in "Offline/Mock Mode" where possible (e.g., `LinguaMaestro`).
- **Remaining:** Full production capabilities require valid `.env` keys.

## 3. Test Suite

**Status:** IMPROVING
- **Update:** `tests/verify_v23_full.py` now verifies the entire Adaptive System loop (Planner, Graph, Self-Correction).
- **Action:** Legacy tests in `tests/` need to be refactored to match the v23 `MetaOrchestrator` pattern.

## 4. UI Backend

**Status:** PENDING
- **Issue:** The `services/webapp` requires a full `npm` build.
- **Workaround:** The system currently relies on "Static Mode" (`showcase/index.html`) using `js/mock_data.js` for demonstration purposes.
