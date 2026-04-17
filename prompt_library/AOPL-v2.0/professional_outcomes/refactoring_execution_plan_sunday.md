# Sunday Execution Plan: The Validator (Comprehensive Test Coverage)

**Objective**: Apply "The Validator" sequentially across all major repository domains over a 12-week timeline. Generate comprehensive unit, integration, and UI tests to lock in progress and ensure system stability.

## Phase 1: Core Engine and Backend Validation (Weeks 1-3)
* **Week 1 (v30 Agent Tests)**: Target `core/v30_architecture/`. Write isolated unit tests for all derived `BaseAgent` classes, ensuring the `execute(**kwargs)` method handles parameters correctly and emits appropriate `NeuralPacket` objects.
* **Week 2 (Engine Factory Tests)**: Target `core/engine/factory.py`. Write integration tests that simulate primary Rust layer (`RealTradingEngine`) failures and assert that the `LiveMockEngine` fallback engages smoothly without data loss.
* **Week 3 (API and Controller Tests)**: Target `backend/`. Use FastAPI's `TestClient` to validate endpoints. Test the dual-layer evaluation harness and ensure the controller's retry loops function as expected under failure conditions.

## Phase 2: Orchestration and Script Validation (Weeks 4-6)
* **Week 4 (Daily Ritual Tests)**: Test `scripts/daily_ritual.py`. Mock the `litellm` responses and assert that the regex-based parser correctly extracts data or gracefully defaults to the `MOCK_PAYLOAD` when necessary.
* **Week 5 (Dashboard Generation Tests)**: Test HTML generation scripts (e.g., `generate_daily_index.py`). Use simple string matching or HTML parsers (like BeautifulSoup) in tests to ensure generated files contain the required JS arrays and masonry structures.
* **Week 6 (Pre-existing Test Audit)**: Review the known broken tests (e.g., `test_v30_market_scanner.py`, `test_new_engines.py`). Decide strategically whether to fix them to align with the new architecture or safely remove them if the underlying code was purged.

## Phase 3: Mock Ecosystem and Graceful Degradation (Weeks 7-9)
* **Week 7 (Static Mock Mode Verification)**: Write integration tests that explicitly set `MOCK_MODE=true` and `ENV=demo`. Verify that all downstream components route requests to the `config/mocks/` directory instead of live APIs or Rust layers.
* **Week 8 (Mock Connector Validation)**: Unit test the mock clients (e.g., `BigQueryConnector`, `LakehouseConnector`). Ensure they return appropriately structured Pydantic models rather than empty text stubs.
* **Week 9 (Dynamic Search Hierarchy)**: Test the fallback mechanisms of the search agents. Simulate primary proxy failures (e.g., EDGAR delays) and assert that the agent successfully falls back to secondary proxies without hallucinating data.

## Phase 4: Frontend UI and E2E Verification (Weeks 10-12)
* **Week 10 (Playwright Setup & Smoke Tests)**: Implement Playwright to verify static HTML frontend changes. Write basic smoke tests that load local files (`file://`) and assert that critical UI elements render without errors.
* **Week 11 (Interactive Workflow Tests)**: Use Playwright to test the 3-stage interactive illiquid market maker dashboard in `MarketMayhem.tsx` (Directory -> Tearsheet -> Pricing Drill-Down).
* **Week 12 (Gated Content & CI Validation)**: Test the gated insider access UI mechanism, ensuring the blur effect toggles correctly upon authentication. Finalize the test suite, ensure clean execution via `uv run pytest`, and integrate seamlessly into the CI pipeline.
