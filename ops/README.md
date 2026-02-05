# Operations & Deployment

This directory contains the infrastructure-as-code and scripts for deploying Adam v26.0.

## 🐳 Docker Deployment

Adam is designed to run as a containerized microservice.

### Structure
*   **`Dockerfile`**: The main application image. Builds the Python environment.
*   **`docker-compose.yml`**: Orchestrates the service mesh.
    *   `adam-core`: The Python backend.
    *   `adam-web`: The React frontend.
    *   `redis`: Message broker for System 1.
    *   `neo4j`: Knowledge Graph.

### Commands
```bash
# Build and Start
docker-compose up --build -d

# View Logs
docker-compose logs -f adam-core

# Stop
docker-compose down
```

## 🛡️ Security Checks

Before deployment, run the security audit script:
```bash
python ops/security/run_checks.py
```
This verifies:
1.  No secrets in code.
2.  Dependencies are pinned.
3.  Permissions are restricted.

## 🔄 CI/CD

We use GitHub Actions for Continuous Integration.
*   **On Push:** Runs unit tests (`pytest`).
*   **On Merge:** Builds Docker image and pushes to registry.
