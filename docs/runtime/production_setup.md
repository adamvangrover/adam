# Adam v26.0 Production Setup Guide

This guide details how to deploy Adam v26.0 in a production-ready environment.

## 1. Environment Strategy

We support three primary deployment modes:
1.  **Bare Metal / VM (High Performance):** For maximum IOPS and GPU access.
2.  **Docker Compose (Standard):** For isolated, reproducible deployments.
3.  **Kubernetes (Scale):** For managing the Swarm across a cluster.

## 2. Dependency Management: `uv`

We strictly use **`uv`** for Python package management. It is significantly faster than pip/poetry and ensures deterministic builds.

### Installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Workflow
*   **Sync:** `uv sync` (Installs everything from `uv.lock`)
*   **Add Package:** `uv pip install package_name` (Then update `pyproject.toml`)
*   **Run Script:** `uv run scripts/run_adam.py`

## 3. Configuration

### 3.1 Secrets (`.env`)
Production deployments **must** use a secrets manager (e.g., Vault, AWS Secrets Manager) to inject environment variables. For local/MVP, use `.env`.

**Critical Keys:**
*   `OPENAI_API_KEY`: For the Planner (GPT-4o recommended).
*   `ANTHROPIC_API_KEY`: For Code Gen (Claude 3.5 Sonnet recommended).
*   `FMP_API_KEY`: Financial Modeling Prep (Market Data).
*   `NEO4J_URI` / `NEO4J_PASSWORD`: Knowledge Graph.
*   `REDIS_URL`: Swarm message bus.

### 3.2 Feature Flags (`config/features.yaml`)
Toggle capabilities without redeploying.
```yaml
features:
  swarm_mode: enabled
  quantum_simulation: disabled
  live_trading: disabled
```

## 4. Docker Deployment

### 4.1 Build
```bash
docker build -t adam-v26-core -f Dockerfile.core .
```

### 4.2 Compose
Use the standard `docker-compose.yml` for a full stack (Core + Redis + Postgres + Neo4j).

```bash
docker-compose up -d
```

**Services:**
*   `adam-core`: The Python backend / MCP Server.
*   `redis`: Message broker for System 1 agents.
*   `neo4j`: Graph database for Entity resolution.
*   `postgres`: Relational store for logs and structured data.

## 5. Monitoring & Observability

*   **Logs:** All logs are structured JSON. Forward to ELK/Splunk/Datadog.
*   **Tracing:** Enable `OTEL_EXPORTER` to send OpenTelemetry traces to Jaeger/Zipkin.
*   **Health Checks:** Monitor `/health` endpoint on the MCP server.

## 6. Scaling the Swarm

To scale System 1 (The Swarm):
1.  Deploy multiple instances of the `adam-swarm-worker` container.
2.  Ensure they share the same `REDIS_URL`.
3.  The `HiveMind` manager will automatically distribute tasks.
