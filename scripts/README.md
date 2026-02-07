# Adam Script Registry

This directory contains the operational tools for running, testing, and maintaining the Adam system.

## 🏃 Execution

*   **`run_adam.py`**: **The Main Entry Point.** Runs the CLI or launches the core engine.
    ```bash
    python scripts/run_adam.py --query "Analyze AAPL"
    ```
*   **`swarm_showcase.py`**: Runs a visual demo of the Swarm agents in action (console animation).

## 🧪 Simulation & Testing

*   **`run_simple_simulation.py`**: A lightweight test of the simulation engine.
*   **`run_llm_driven_simulation.py`**: Launches a complex, multi-agent scenario driven by LLMs.
*   **`benchmark_adam.py`**: Measures throughput and latency of the Knowledge Graph.

## 🛠️ Data Generation

*   **`generate_ui_data.py`**: Creates mock JSON data for the frontend dashboard (useful for offline dev).
*   **`generate_market_mayhem_archive.py`**: Generates historical scenarios for the "Market Mayhem" game.
*   **`fetch_market_data.py`**: Connects to external APIs (FMP, Yahoo) to populate the local DB.

## 🧹 Maintenance

*   **`initialize_comprehensive_memory.py`**: Resets and seeds the vector database.
*   **`archive_ui_artifacts.py`**: Backs up generated reports to the `archive/` folder.

## ⚠️ Important Note
Always run scripts from the **repository root** to ensure imports work correctly:
```bash
export PYTHONPATH=.
python scripts/script_name.py
```
