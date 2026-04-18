# Engine Factory

The Engine Factory (`core/engine/factory.py`) provides the core logic to rotate runtime environments seamlessly between Simulation, Live, and Backtest modes.

## Graceful Fallback Mechanism

The Factory pattern ensures that if the primary Rust layer (`RealTradingEngine`) is unavailable or fails, it gracefully falls back to the Python-based `LiveMockEngine`. This ensures the system remains operational even during degraded states.

## PyO3 Bindings

The Engine interfaces with the Rust Execution layer (`core/rust_pricing/`) via PyO3, allowing computationally intensive tasks like market pricing to run natively in Rust while maintaining high-level orchestration in Python.

## Data Handoff Structures

Data handoffs between Python and Rust use native mapping functions (e.g., passing tuples and structs), ensuring type safety and minimal serialization overhead.
