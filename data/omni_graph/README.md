# The Adam Omni-Graph
# Adam Omni-Graph

This directory contains the data layer for Adam v23.5.

## Structure
- **constellations/**: Tier 1 - Breadth (Sector-wide light data)
- **dossiers/**: Tier 2 - Depth (Deep Dive Profiles)
- **templates/**: Tier 3 - Archetypes (Abstract templates)
- **relationships/**: The Edges (Supply Chain, Competitors)

# The Adam Omni-Graph (v23.5 Data Layer)

**Strategic Goal:** Move Adam v23.5 from a "Proof of Concept" to a "Platform" by establishing a "Golden Source" Universe.

This structured Data Layer acts as the system's "Knowledge Graph," providing a rich library of pre-computed profiles and relationships. This allows the UI to look densely populated (breadth) while enabling deep simulations on specific entities (depth).

## Tiered Data Architecture

The data is organized into three tiers to balance performance and depth:

### Tier 1: The Constellations (Breadth)
*   **Purpose:** Populate the visual graph (e.g., 3D webapp) with lightweight nodes.
*   **Content:** Name, Ticker, Sector, Market Cap, Relationship to Hero.
*   **Location:** `constellations/`

### Tier 2: The Hero Dossiers (Depth)
*   **Purpose:** Serve as "Ground Truth" for "Deep Dive" demonstrations. These are full v23.5 Schema-compliant JSONs that act as pre-computed runtime results.
*   **Content:** Full entity ecosystem, equity analysis, credit analysis, simulation engine results.
*   **Location:** `dossiers/`

### Tier 3: The Archetypes (Templates)
*   **Purpose:** Instantly generate new, realistic dummy companies for simulations.
*   **Content:** Abstract templates (e.g., "SaaS_Growth_HighBeta") with defined financial profiles and risk factors.
*   **Location:** `templates/`

## Relationships (Edges)
*   **Purpose:** Define the connections between entities (e.g., supply chains, competitors).
*   **Location:** `relationships/`

## Usage

Use the `scripts/load_omni_graph.py` utility to ingest these JSON files into a NetworkX or Neo4j graph object at runtime.
