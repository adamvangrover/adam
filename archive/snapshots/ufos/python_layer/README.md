# The Ghost in the Machine

This directory contains the Python-based "Intelligence Layer" of the Unified Financial Operating System (UFOS).

## Components

### 1. Agent (`agent.py`)
*   **Role**: The primary "Ghost" that oversees the trading engine.
*   **Protocol**: Acts as an MCP Client, connecting to the Rust Core via SSE/HTTP.
*   **Function**: Consumes market data resources and invokes execution tools.

### 2. Strategy Tuner (`strategy_tuner.py`)
*   **Role**: Reinforcement Learning (RL) agent for parameter optimization.
*   **Algorithm**: Proximal Policy Optimization (PPO).
*   **Targets**: Dynamically tunes $\gamma$ (Risk Aversion) and $\kappa$ (Arrival Intensity) in the Avellaneda-Stoikov model running in the Rust core.

### 3. Anomaly Detection (Planned)
*   **Library**: `htm.core` (Hierarchical Temporal Memory).
*   **Purpose**: Detects microstructure anomalies (e.g., spoofing, flash crashes) in real-time SDR format.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Agent

Ensure the Rust Core is running on port 3000.

```bash
python agent.py
```
