# Contributing to Adam v23.0

Thank you for your interest in contributing to Adam v23.0 ("The Adaptive Hive Mind")! We welcome contributions to enhance this advanced financial analytics system.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 16+ (for frontend)
- Docker (optional, for containerized run)

### Installation
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
3.  **Install Frontend Dependencies (if working on UI):**
    ```bash
    cd services/webapp/client
    npm install --legacy-peer-deps
    ```

## 🧪 Running Tests

We use `pytest` for backend testing. Ensure you set `PYTHONPATH=.` when running tests from the root.

```bash
PYTHONPATH=. python -m pytest tests/test_agent_orchestrator.py
PYTHONPATH=. python -m pytest tests/test_v23_5_pipeline.py
```

## 📂 Project Structure

- **core/**: The brain of the system. Contains agents, engines, and shared logic.
    - **core/agents/**: Agent definitions (e.g., `RiskAssessmentAgent`).
    - **core/engine/**: Graph engines (`MetaOrchestrator`, `NeuroSymbolicPlanner`).
    - **core/schemas/**: Pydantic data models.
- **tests/**: Unit and integration tests.
- **services/webapp/**: The React-based frontend and Flask API.
- **showcase/**: Static HTML visualizers.
- **docs/**: Documentation.

## 🤝 Contribution Workflow

1.  **Pick a Task:** Check `AGENTS.md` or open issues.
2.  **Branch:** Create a feature branch (`feat/your-feature`) or bugfix branch (`fix/issue-desc`).
3.  **Code:** Implement changes. Follow PEP 8.
4.  **Test:** Add unit tests in `tests/` and verify they pass.
5.  **Submit PR:** Open a Pull Request with a clear description.

### Code Style
- Use **Pydantic v2** for data models.
- Use **AsyncIO** for I/O bound tasks.
- Use **LangGraph** for complex agent flows.

## 🐛 Reporting Bugs
Please use the GitHub Issue tracker and include:
- Steps to reproduce
- Expected behavior
- Stack traces or logs

Thank you for building the future of autonomous finance!
