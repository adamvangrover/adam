# Wednesday Execution Plan: The Optimizer (Performance & Security Hardening)

**Objective**: Apply "The Optimizer" sequentially across all major repository domains over a 12-week timeline. Audit for bottlenecks, enforce resource safety, and secure the system against vulnerabilities.

## Phase 1: Security and Network Safety (Weeks 1-3)
* **Week 1 (Backend Security)**: Run `uv run bandit -r core services scripts -ll -x tests`. Resolve legitimate security warnings (e.g., B104 binding to `0.0.0.0` by restricting hosts to `127.0.0.1`). Avoid lazy `# nosec` suppressions.
* **Week 2 (Dependency Auditing)**: Audit `pyproject.toml` and lockfiles for known vulnerabilities. Update packages via `uv` while ensuring compatibility with the legacy `docker-compose.yml` deployment pathways.
* **Week 3 (Access Control & Gating)**: Review frontend UI mechanisms (e.g., the authentication modal `#authModal` in `showcase/adam_daily_hub.html`). Ensure gated insider access logic is securely implemented on the backend, not just visually blurred on the frontend.

## Phase 2: Backend and Execution Efficiency (Weeks 4-6)
* **Week 4 (Async Profiling)**: Profile `backend/api.py` and the v30 swarm runner. Identify blocking synchronous operations within the event loop and convert them to `async`/`await` for higher throughput.
* **Week 5 (Memory Management)**: Audit long-running evaluation loops in `backend/controller.py`. Implement explicit resource cleanup (closing database/mock client connections) and prevent memory leaks from unreferenced tasks.
* **Week 6 (Rust Execution Tuning)**: Profile the `core/rust_pricing/` execution layer. Optimize PyO3 data transfer structures to minimize serialization/deserialization overhead between Python and Rust.

## Phase 3: Frontend and Data Rendering Optimization (Weeks 7-9)
* **Week 7 (React Render Cycles)**: Profile `services/webapp/client`. Identify redundant re-renders in heavy components like `PromptAlpha.tsx`. Implement `React.memo` and optimize sorting operations (e.g., using `a.timestamp - b.timestamp` directly).
* **Week 8 (Dashboard Generation Speed)**: Profile large scripts like `scripts/generate_daily_index.py`. Optimize regex parsing of JavaScript arrays in HTML files and implement parallel file processing to speed up dashboard generation.
* **Week 9 (Asset Delivery)**: Audit `showcase/` HTML dashboards. Ensure heavy assets (like machine learning JSON blobs or large images) are loaded lazily or compressed to improve terminal rendering speed.

## Phase 4: Reliability and Graceful Fallbacks (Weeks 10-12)
* **Week 10 (Controller Resilience)**: Optimize the self-healing CLI controller (`backend/controller.py`). Implement exponential backoff for retry loops to prevent hammering mocked APIs or live endpoints during outages.
* **Week 11 (Fallback Efficiency)**: Test the `MOCK_MODE=true` toggle. Ensure the transition to lightweight Python static proxies (`config/mocks/`) is instantaneous and does not incur unnecessary timeout delays from the primary Rust layer.
* **Week 12 (Stress Testing)**: Simulate high-load concurrent requests against the optimized backend APIs. Monitor memory usage and ensure the application fails gracefully without crashing the core engine.
