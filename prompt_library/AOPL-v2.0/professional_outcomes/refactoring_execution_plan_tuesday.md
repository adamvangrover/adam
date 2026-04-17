# Tuesday Execution Plan: The Refactor (Global Modularity Initiative)

**Objective**: Apply "The Refactor" sequentially across all major repository domains over a 12-week timeline. Reorganize code for high modularity, strict architectural boundaries, and adherence to modern best practices.

## Phase 1: Backend and Execution Routing (Weeks 1-3)
* **Week 1 (API & Core Decoupling)**: Refactor `backend/api.py`. Extract complex business logic into dedicated service layers (`backend/services/`), leaving the API layer strictly for HTTP routing and Pydantic validation.
* **Week 2 (Engine Factory Design)**: Review `core/engine/factory.py`. Strengthen the boundary and Dependency Injection patterns between the `RealTradingEngine` (Rust) and the `LiveMockEngine` (Python fallback).
* **Week 3 (Swarm Orchestration)**: Refactor `core/v30_architecture/python_intelligence/agents/swarm_runner.py`. Ensure asynchronous task management uses strong references and scalable Event-Driven patterns (e.g., LangGraph or queue structures).

## Phase 2: System Interfaces and Connectors (Weeks 4-6)
* **Week 4 (Mock Institutional Infrastructure)**: Refactor mock data providers. Ensure strict Dependency Injection is used to allow seamless switching between live APIs and the static Python proxies in `config/mocks/`.
* **Week 5 (Rust/Python Bridge)**: Review the PyO3 integration in `core/rust_pricing/`. Refactor the interface to ensure clean, typed data handoffs between the Python backend and the computationally heavy Rust layer.
* **Week 6 (Data Ingestion Modularity)**: Refactor data ingestion scripts in `scripts/`. Abstract repetitive file parsing logic into a shared `utils/data_parser.py` module to service multiple generation pipelines.

## Phase 3: Frontend Architecture (Weeks 7-9)
* **Week 7 (React Component Abstraction)**: Target `services/webapp/client/`. Abstract repetitive UI elements (e.g., masonry cards in `MarketMayhem.tsx`) into reusable, styled sub-components.
* **Week 8 (Type Contract Alignment)**: Enforce strict contract boundaries. Refactor TypeScript interfaces to perfectly mirror backend Pydantic models. Establish a shared schema definition process.
* **Week 9 (State Management & Re-renders)**: Optimize frontend state. Refactor array stream filtering (e.g., hoisting `.toLowerCase()` outside `.filter()` loops) to prevent massive string allocation bottlenecks during React re-renders.

## Phase 4: Prompts, Tests, and Standardization (Weeks 10-12)
* **Week 10 (Prompt Hierarchy Structuring)**: Refactor the Adam Operational Prompt Library (`prompt_library/AOPL-v2.0/`). Organize domain-specific analytical prompts strictly into targeted subdirectories (e.g., `professional_outcomes/`).
* **Week 11 (Test Modularity)**: Refactor `tests/`. Abstract repetitive setup logic into reusable `pytest` fixtures. Ensure test files are clearly separated by domain (unit, integration, mock-fallback validation).
* **Week 12 (Global Idiom Alignment)**: Conduct a final cross-domain sweep. Ensure consistent absolute import structures (banning `sys.path.append`) and adherence to single-source-of-truth configuration via `pyproject.toml`.
