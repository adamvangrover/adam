# Unified Financial Operating System (UFOS)

> "The Iron Core and the Ghost in the Machine"

This repository implements the architectural blueprint for a hybrid low-latency trading system and agentic AI.

## Directory Structure

*   **`rust_core/` (The Iron Core)**: The microsecond-latency execution engine written in Rust. Handles order matching, risk checks, and MCP server duties.
*   **`python_layer/` (The Ghost)**: The probabilistic AI layer. Handles strategy tuning, anomaly detection, and agentic reasoning via MCP.

## Architecture

1.  **Rust Core**: Zero-allocation, lock-free, `io_uring` networking, Avellaneda-Stoikov strategy.
2.  **MCP**: Model Context Protocol acts as the standardized membrane between Rust (Server) and Python (Client).
3.  **Python Layer**: PPO agents and HTM anomaly detectors.

## Quick Start

1.  Start the Iron Core:
    ```bash
    cd rust_core
    cargo run --release
    ```

2.  Start the Ghost Agent:
    ```bash
    cd python_layer
    pip install -r requirements.txt
    python agent.py
    ```
