# Production Setup Guide

## Prerequisites
*   **Docker & Docker Compose**: For containerized deployment.
*   **Kubernetes (Optional)**: For scaling agent swarms.
*   **Redis**: For high-speed message bus and caching.
*   **PostgreSQL**: For transactional state and audit logs.

## Deployment Steps

### 1. Configuration
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY=sk-...
# DATABASE_URL=postgresql://user:pass@localhost:5432/adam
# REDIS_URL=redis://localhost:6379/0
```

### 2. Docker Compose (Recommended for Single Node)
```bash
docker-compose up -d --build
```
This spins up:
- `adam-core`: The main intelligence API.
- `adam-worker`: Async background workers.
- `postgres`: Database.
- `redis`: Message broker.

### 3. Kubernetes (Enterprise Scale)
Apply the manifests in `k8s/`:
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Security Hardening
*   **Network Isolation**: Ensure the `Intelligence Layer` cannot access the `Execution Layer` directly without passing through the `Governance Gatekeeper`.
*   **Least Privilege**: Agents run with scoped API keys (e.g., `READ_ONLY` for market data).
*   **Audit Logging**: Enable `PROVENANCE_LOGGING=true` to track every decision trace.

## Monitoring
*   **Prometheus/Grafana**: Metrics are exposed at `/metrics`.
*   **Logfire**: Structured logging for agent reasoning traces.
