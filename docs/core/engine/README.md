# Engine Layer and Execution Factory

The `core/engine/` directory manages the boundary between the high-level Python multi-agent swarm and the low-level execution engines.

## `EngineFactory` (`factory.py`)
The `EngineFactory` implements a factory pattern for runtime environment rotation, allowing the system to seamlessly switch between different execution modes:

- **SIMULATION (`LiveMockEngine`)**: Engaged when the environment is set to `SIMULATION` or `MOCK_MODE=true`. This is a Python-based singleton engine used for safe, static, or isolated testing, preventing unintended side effects.
- **LIVE / PRODUCTION (`RealTradingEngine`)**: The primary execution layer for real-world interactions and high-stakes computational tasks.

### Graceful Fallback Mechanism
The boundary guarded by `EngineFactory` is designed with robust graceful degradation. If the primary Rust layer (`RealTradingEngine`) fails to initialize, encounters a connection error, or is running in an environment without the compiled Rust binaries, the system will fall back to the Python-based `LiveMockEngine` to maintain operational continuity.

## Rust Execution Layer (`core/rust_pricing/`)
Computationally intensive tasks, such as high-frequency market pricing and large-scale quantitative matrix operations, are offloaded to a high-performance Rust execution layer.

- **PyO3 Bindings**: The Rust codebase is exposed to the Python backend via `pyo3` bindings, allowing Python agents to call Rust methods directly.
- **Data Handoff**: Data is passed from the Python swarm into the Rust layer using typed, structured payloads to ensure memory safety and zero-cost abstractions where possible.
