# Rust Execution Layer

This directory contains the high-performance Rust execution layer, exposed to the Python backend via PyO3.

## Purpose

Handles computationally intensive tasks such as algorithmic market pricing, simulating market mechanics, and executing deterministic trading signals.

## PyO3 Integration & Graceful Fallback

The library exposes standard Rust functions directly to Python via PyO3, abstracting complex Rust structures behind standard Python interfaces.

### Graceful Fallback Mechanism

The Python `EngineFactory` (`core/engine/factory.py`) implements a robust graceful fallback mechanism. If the primary Rust layer (accessed via `RealTradingEngine`) fails to load, crashes, or is not selected in the runtime environment, the system automatically falls back to a Python-based `LiveMockEngine`. This guarantees system continuity without catastrophic failure.

### Data Handoff Structures

Data handoffs between Python and Rust typically use standard primitive types (e.g., floats for pricing models) rather than complex objects to minimize serialization overhead. Future enhancements may utilize Apache Arrow or similarly zero-copy formats for bulk data transfers.

**Example Python usage:**
```python
import rust_pricing
# returns bid/ask tuple
bid, ask = rust_pricing.get_quotes(100.0, 0.15, 0.0, 1.5, 1.0, 0.05)
```
