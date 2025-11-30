# The Adam Omni-Graph

This directory contains the "Golden Source" Universe for Adam v23.5. It is a structured Data Layer designed to provide both breadth and depth for the system's knowledge graph.

## Structure

*   **`constellations/` (Tier 1: Breadth):** Lightweight nodes (Name, Ticker, Sector, Market Cap) to populate the visual graph.
*   **`dossiers/` (Tier 2: Depth):** Full v23.5 Schema-compliant JSONs for key demo companies (e.g., NVDA). These act as pre-computed "runtime results" for Deep Dive agents.
*   **`templates/` (Tier 3: Archetypes):** Abstract templates (e.g., "SaaS_Growth_HighBeta") to instantly generate new, realistic companies for simulations.
*   **`relationships/`:** Edge definitions such as supply chains and competitor links.

## Usage

These files are ingested by the `OmniGraphLoader` script (`scripts/load_omni_graph.py`) to construct the runtime knowledge graph or populate the UI visualization.
