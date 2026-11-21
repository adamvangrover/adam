# SNAPSHOT: ufos

**Purpose:** Unified Financial OS prototype — Rust low-latency core (Avellaneda–Stoikov market making, MCP server) + Python PPO strategy tuner / anomaly agent.

**Notable implementations**
- `rust_core/` — standalone Rust crate; `cargo build` inside it still works independently.
- `python_layer/strategy_tuner.py`, `agent.py` — PPO tuning loop over the Rust core via MCP.

**Pulled:** nothing. Live equivalents: `core/schemas/unified_ledger.sql` (UFOS ledger), `core/rust_pricing/`, `rust_ext/`.

**Not imported** by live code.
