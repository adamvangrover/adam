# 01 - Local Deployment of ADAM

This tutorial will guide you through setting up your local environment for the Autonomous Deterministic Alpha Matrix (ADAM), running the Docker/Kubernetes orchestration, and verifying that the Rust core is communicating with the Python sidecars.

## Prerequisites
- Docker & Docker Compose
- Kubernetes (Minikube or equivalent)
- `uv` Python package manager

## Step 1: Spin up the Orchestration
First, set your required environment variables.
```bash
export OPENAI_API_KEY="sk-..."
export ADAM_LOCAL_DEPLOY=true
```

Start the containers using Docker Compose:
```bash
docker-compose -f docker-compose.yml up -d
```
This spins up the AdamOS kernel and sidecar agents.

## Step 2: Deploy to Kubernetes (Optional but Recommended)
For a robust setup, apply the Kubernetes manifests:
```bash
kubectl apply -f kubernetes/
```

## Step 3: Verify Core Communication
Ensure the Python sidecars are successfully talking to the Rust execution layer. You can do this by pinging the orchestration server.

```bash
uv run python scripts/verify_communication.py
```
You should see a message confirming: "System 2 Rust Kernel is online and connected to System 1 Swarm."

Congratulations! You have successfully deployed ADAM locally.
