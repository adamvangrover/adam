# Deployment Checklist

1.  **Environment Setup**
    *   Ensure Python 3.12+ is installed.
    *   Install dependencies:
        ```bash
        pip install -r ops/requirements.txt
        ```
        or using uv:
        ```bash
        uv pip install -r ops/requirements.txt
        ```

2.  **Configuration**
    *   Set environment variables in `.env` or system:
        *   `FLASK_DEBUG=False` (Production)
        *   `OPENAI_API_KEY` (Required for agents)
        *   `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (For Knowledge Graph)

3.  **Verification**
    *   Run unit tests:
        ```bash
        PYTHONPATH=. python3 -m pytest tests/
        ```
    *   Verify linting:
        ```bash
        flake8 core/ services/
        ```

4.  **Execution**
    *   Start the API server:
        ```bash
        python3 services/webapp/api.py
        ```
    *   Or use the runner script:
        ```bash
        ./scripts/run_adam.py
        ```

5.  **Frontend**
    *   Build React app (if applicable) in `services/webapp/client`:
        ```bash
        npm install --legacy-peer-deps
        npm run build
        ```
