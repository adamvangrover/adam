# Deprecation Notice

This directory (`core/v23_graph_engine`) contains the legacy v23 "Adaptive System" graph engine components.

**Status:** Deprecated but Preserved for Backward Compatibility.

## Migration Guide
The core logic has been migrated to `core/engine/` to consolidate the architecture.
*   `unified_knowledge_graph.py` -> `core/engine/unified_knowledge_graph.py`
*   `simulation_engine.py` -> `core/engine/simulation_engine.py`

## Usage
To use these components, import explicitly from this namespace. The `MetaOrchestrator` in `core/engine/` is capable of routing to these legacy graphs if the system configuration specifies `execution_mode: "legacy_v23"`.
