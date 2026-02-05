# Adam Testing Strategy

Ensuring the reliability of a "Financial Sovereign" requires a rigorous testing pyramid.

## ⚠️ Important Note
Always run tests from the **repository root** with `PYTHONPATH=.`.

```bash
export PYTHONPATH=.
```

## 1. Unit Tests (`tests/test_*.py`)
Fast, isolated tests for individual components.
*   **Target:** `core/agents`, `core/utils`, `core/data_processing`.
*   **Run Command:**
    ```bash
    pytest tests/test_financial_modeling_agent.py
    ```

## 2. Integration Tests (`tests/test_v23_*.py`)
Tests that verify how multiple components work together (e.g., Planner + Agent).
*   **Target:** `core/engine`.
*   **Run Command:**
    ```bash
    pytest tests/test_v23_5_pipeline.py
    ```

## 3. Verification Scripts (`tests/verify_*.py`)
End-to-end "Smoke Tests" that mimic real user behavior. These scripts often spin up the full engine.
*   **Target:** Full system flows.
*   **Examples:**
    *   `verify_deep_dive.py`: runs a full deep dive analysis.
    *   `verify_frontend_war_room.py`: Checks if the frontend APIs are responding.
*   **Run Command:**
    ```bash
    python tests/verify_deep_dive.py
    ```

## 🧪 Testing Standards

*   **Mock External APIs:** Use `unittest.mock` to avoid hitting real OpenAI or Financial APIs during unit tests.
*   **Deterministic:** Tests should pass 100% of the time. Avoid using live LLM calls in the CI pipeline; use "Mock LLMs" instead.
*   **Coverage:** We aim for high coverage in `core/engine` (the logic center) and `core/risk` (the safety center).
