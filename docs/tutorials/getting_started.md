# Tutorial: Getting Started with ADAM v26.0

Welcome to the Autonomous Deterministic Alpha Matrix (ADAM). This tutorial will guide you through setting up your first Swarm agent and executing a deterministic credit risk evaluation.

## Prerequisites
- Python 3.10+
- `uv` package manager installed
- Open AI API Key configured

## Step 1: Environment Setup
First, sync your dependencies using `uv`:
```bash
uv sync
source .venv/bin/activate
```

## Step 2: Running the Pulse Simulator
ADAM operates via the Orchestrator Engine. Run the pulse simulator to verify your environment:
```bash
uv run python scripts/run_adam.py
```
This will initialize the System 1 Swarm and the System 2 Neuro-Symbolic Graph.

## Step 3: Triggering a Credit Sentinel Analysis
You can prompt the Credit Sentinel to analyze a mock 10-K document. The output will be parsed and evaluated against deterministic JSONLogic rules.

By completing this tutorial, you now have a foundational understanding of deploying ADAM locally.
