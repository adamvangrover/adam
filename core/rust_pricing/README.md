# Rust Execution Layer

This directory contains the high-performance Rust execution layer, exposed to the Python backend via PyO3.

## Purpose

Handles computationally intensive tasks such as algorithmic market pricing, simulating market mechanics, and executing deterministic trading signals.

## PyO3 Integration

The library exposes standard Rust functions directly to Python.

**Example Python usage:**
```python
import rust_pricing
# returns bid/ask tuple
bid, ask = rust_pricing.get_quotes(100.0, 0.15, 0.0, 1.5, 1.0, 0.05)
```
