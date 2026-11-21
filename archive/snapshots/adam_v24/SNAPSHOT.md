# SNAPSHOT: adam_v24

**Purpose:** v24 "Unified Financial OS" blueprint — Rust Iron Core (order matching), Python intelligence layer (HNASP, MCP server, evolutionary architect), React cockpit.

**Notable implementations**
- `iron_core/` — Rust order-book / risk-check prototype.
- `intelligence_layer/hnasp/`, `schemas.py` — early HNASP state protocol (superseded by `core/schemas/hnasp*.py`).
- `intelligence_layer/mcp_server.py` — early MCP tool bus (superseded by `server/server.py`).
- `tests/test_evolutionary.py` — AST-mutation "Gauntlet" test idea.

**Pulled:** `schemas/unified_ledger.sql` → `data/schemas/unified_ledger_bitemporal.sql` (bitemporal order ledger + vector context table; distinct from `core/schemas/unified_ledger.sql`).

**Not imported** by live code. Learning reference only.
