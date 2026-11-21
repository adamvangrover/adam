# Unified Financial Operating System (UFOS) - Rust Core

This directory contains the "Iron Core" implementation of the UFOS, a microsecond-latency trading engine built in Rust.

## Architecture

Based on "The Iron Core and the Ghost in the Machine", the system is architected as follows:

### 1. The Iron Core (Rust)
*   **Memory**: Zero-allocation architecture using `Arena<T>` and Slab allocation to prevent allocator pause.
*   **Execution**: Lock-free Order Book based on the LMAX Disruptor pattern (simplified with Ring Buffers).
*   **Data Layout**: Data-Oriented Design (DOD) with `#[repr(C)]` and Cache Line alignment `#[repr(align(64))]`.
*   **Networking**: Architecture for `io_uring` kernel bypass.
*   **Strategy**: Avellaneda-Stoikov market making model with atomic parameter updates.
*   **Risk**: Hardware-level kill switch semantics and pre-trade risk checks.

### 2. The Ghost in the Machine (MCP)
*   **Protocol**: Model Context Protocol (MCP) server implementation.
*   **Transport**: Supports SSE (Server-Sent Events) over HTTP via `axum`.
*   **Interface**: Exposes market data as *Resources* and execution capabilities as *Tools*.
*   **Cognitive Bridge**: `PyO3` bindings for zero-copy data exchange with Python-based AI agents.

## Directory Structure

*   `src/memory`: Memory management (Arena, Slab).
*   `src/execution`: Order Book, Matching Engine.
*   `src/strategy`: Quantitative models (Avellaneda-Stoikov).
*   `src/mcp`: MCP Server, Tools, Resources.
*   `src/risk`: Risk controls.
*   `src/networking`: Low-level I/O.
*   `src/bridge`: Python bindings.

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

## Python Integration

The core is exposed as a Python module `ufos_core` via PyO3.

```python
import ufos_core
bridge = ufos_core.UfosBridge()
bridge.update_parameters(gamma=0.5, kappa=10.0)
```
