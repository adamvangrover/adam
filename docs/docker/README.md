# Docker Deployment Ecosystem

The repository supports multiple distinct, redundant deployment pathways orchestrated primarily by `docker/docker-compose.yml` (or the root `docker-compose.yml`).

## Deployment Pathways

1. **`core-engine-legacy`**: Utilizes `Dockerfile.core`. This pathway supports the classic backend services (Flask API, Celery Workers, Postgres, Qdrant, TimescaleDB, Neo4j) for robust operational needs.
2. **`swarm-engine`**: Utilizes `Dockerfile.swarm`. Focuses on the standalone agent mesh execution environment.
3. **`modern-engine`**: Utilizes `Dockerfile.modern` (and `v24-dashboard` via `services/v24_dashboard`). This path runs the modern React client (`services/webapp/client`) alongside the Rust execution layer.

## How to Invoke Locally

To spin up a specific pathway, ensure Docker is running and execute:
```bash
# Example for full standard stack
docker-compose up --build

# Example for specific services
docker-compose up client api db
```
*(Always ensure environment variables via `.env` are configured correctly before invocation.)*
