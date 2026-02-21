# AdamOS Kernel (Prototype)

This is the Rust-based micro-kernel for Adam v25.0 ("Project OMEGA").

## Architecture
- **Kernel**: Handles message routing, memory safety, and agent lifecycle.
- **Agents**: Run as WASM modules (simulated here as structs).
- **Communication**: Zero-copy shared memory via Apache Arrow (planned).

## Usage
This is currently a prototype library.
To run tests (if Rust is installed):
```bash
cargo test
```

## Migration Plan
1.  Port `MetaOrchestrator` logic to `src/lib.rs`.
2.  Implement `WasmRuntime` struct to load `.wasm` files.
3.  Replace Python `AgentOrchestrator` with PyO3 bindings to this crate.
