# Deployment Guide

This guide covers the deployment strategies for the Adam platform, including local development, Docker containers, and cloud environments.

## 1. Local Development

### Prerequisites
- Python 3.10 or higher (3.12 recommended)
- Node.js & npm (for frontend)
- Redis (optional, for async tasks)
- Neo4j (optional, for knowledge graph)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/adam.git
    cd adam
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    # Run the main script
    python scripts/run_adam.py

    # Or run the Web API (Flask)
    python services/webapp/api.py
    ```

## 2. Docker Deployment

The project includes `Dockerfile` for the main web application and `Dockerfile.core` for the core API service.

### Using Docker Compose (Recommended)

The `docker-compose.yml` orchestrates the services (Webapp, Redis, Neo4j).

1.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```

2.  **Access:**
    - Web Interface: `http://localhost:3000` (or `http://localhost:5001` for API)
    - Neo4j Browser: `http://localhost:7474`

### Manual Docker Build

**Main Web Application:**
```bash
docker build -t adam-webapp -f Dockerfile .
docker run -p 5001:5001 adam-webapp
```

**Core API:**
```bash
docker build -t adam-core -f Dockerfile.core .
docker run -p 8000:8000 adam-core
```

## 3. Cloud Deployment

### Google Cloud Run / AWS Fargate

Since the application is containerized, it can be deployed to serverless container platforms.

1.  **Build the image:**
    ```bash
    gcloud builds submit --tag gcr.io/your-project/adam-webapp
    ```

2.  **Deploy:**
    Ensure you set the necessary environment variables (`OPENAI_API_KEY`, `NEO4J_URI`, `REDIS_URL`) in your cloud console.

### Kubernetes

For scalable production deployment:
1.  Use the `Dockerfile` to build your images.
2.  Deploy `redis` and `neo4j` as stateful sets (or use managed services).
3.  Deploy `adam-webapp` and `adam-core` as deployments/services.

## 4. Verification

To verify the deployment is working:
1.  Check the health endpoint: `GET /api/hello` or `/health`.
2.  Run the validation script:
    ```bash
    python tests/validate_ukg_seed.py
    ```
