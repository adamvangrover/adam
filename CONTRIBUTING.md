# Contributing to Adam v23.5

Thank you for your interest in contributing to Adam v23.5 ("The Adaptive Hive Mind")! We welcome contributions to enhance this advanced financial analytics system.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 16+ (for frontend)
- Docker (optional, for containerized run)
- `uv` (Recommended for Python package management)

### Installation

#### Using `uv` (Recommended)
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    ```
2.  **Sync dependencies:**
    ```bash
    uv sync
    source .venv/bin/activate
    ```

#### Using `pip`
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

### Frontend Setup
1.  **Install Dependencies:**
    ```bash
    cd services/webapp/client
    npm install --legacy-peer-deps
    ```

## 🧪 Running Tests

We use `pytest` for backend testing. Ensure you set `PYTHONPATH=.` when running tests from the root.

```bash
# Run all tests
PYTHONPATH=. pytest tests/

# Run specific integration tests
PYTHONPATH=. python -m pytest tests/test_agent_orchestrator.py
PYTHONPATH=. python -m pytest tests/test_v23_5_pipeline.py
```

## 📂 Project Structure

- **core/**: The brain of the system. Contains agents, engines, and shared logic.
    - **core/agents/**: Agent definitions (e.g., `RiskAssessmentAgent`).
    - **core/engine/**: Graph engines (`MetaOrchestrator`, `NeuroSymbolicPlanner`).
    - **core/system/v22_async/**: Asynchronous message-passing infrastructure.
    - **core/data_processing/**: Ingestion and ETL pipelines.
    - **core/schemas/**: Pydantic data models.
- **tests/**: Unit and integration tests.
- **services/webapp/**: The React-based frontend and Flask API.
- **showcase/**: Static HTML visualizers.
- **docs/**: Documentation.

## 🎨 Coding Standards

To maintain the quality and reliability of the "Financial Sovereign", please adhere to the following standards:

### Python
- **Type Hinting:** Use strict type hints for all function signatures. Use `typing` module or native types.
- **Pydantic v2:** Use Pydantic models for all data schemas and agent IO.
- **AsyncIO:** Use `async/await` for all I/O bound tasks (API calls, DB access).
- **Docstrings:** Include Google-style docstrings for all classes and public methods.
- **Error Handling:** Use specific exception types and ensure errors are logged via `core.utils.logging_utils`.

### Architecture
- **Hybrid Model:** Respect the distinction between the **Graph Engine** (synchronous/stateful reasoning) and the **Async Swarm** (stateless/parallel execution).
- **LangGraph:** Use `LangGraph` for complex, multi-step agent workflows that require state persistence.

## 🤝 Contribution Workflow

1.  **Pick a Task:** Check `AGENTS.md` or open issues.
2.  **Branch:** Create a feature branch (`feat/your-feature`) or bugfix branch (`fix/issue-desc`).
3.  **Code:** Implement changes. Follow PEP 8.
4.  **Test:** Add unit tests in `tests/` and verify they pass.
5.  **Submit PR:** Open a Pull Request with a clear description.

## 🐛 Reporting Bugs
Please use the GitHub Issue tracker and include:
- Steps to reproduce
- Expected behavior
- Stack traces or logs

Thank you for building the future of autonomous finance!
