# Adam Omni-Graph

This directory contains the data layer for Adam v23.5.

## Structure
- **constellations/**: Tier 1 - Breadth (Sector-wide light data)
- **dossiers/**: Tier 2 - Depth (Deep Dive Profiles)
- **templates/**: Tier 3 - Archetypes (Abstract templates)
- **relationships/**: The Edges (Supply Chain, Competitors)

# The Adam Omni-Graph (v23.5 Data Layer)

## Strategic Overview
The Adam Omni-Graph is the "Golden Source" universe for the Adam v23.5 platform. It replaces simple seed data with a structured, tiered architecture designed to support both broad market visualization and deep, specific simulations.

## Tiered Architecture

### Tier 1: The Constellations (Breadth)
*   **Location:** `constellations/`
*   **Purpose:** Lightweight nodes to populate the visual graph (e.g., in a 3D web application).
*   **Content:** Minimal viable data: ID, Name, Sector, Role, Market Cap, Relationship to Hero.
*   **Use Case:** Providing context and visual density around the primary "Hero" targets.

### Tier 2: The Hero Dossiers (Depth)
*   **Location:** `dossiers/`
*   **Purpose:** Fully fleshed-out, schema-compliant JSONs that represent a "perfect run" of the Deep Dive analysis.
*   **Content:** Comprehensive data matching `v23_5_schema.py`: Entity Ecosystem, Equity Analysis, Credit Analysis, Simulation Engine, Strategic Synthesis.
*   **Use Case:** Instant, high-fidelity demos of the system's capabilities without incurring the latency or cost of live agent execution.

### Tier 3: The Archetypes (Templates)
*   **Location:** `templates/`
*   **Purpose:** Abstract templates for generating synthetic entities.
*   **Content:** Patterns for naming, financial profiles, risk factors, and default scenarios.
*   **Use Case:** Enabling "what-if" simulations on hypothetical companies (e.g., "Simulate a distressed retail chain in a high-interest rate environment").

### Relationships (The Edges)
*   **Location:** `relationships/`
*   **Purpose:** Defining the connectivity between nodes.
*   **Content:** Supply chain links, competitor mappings, customer relationships.
*   **Use Case:** Modeling second and third-order effects (e.g., "How does a strike at TSMC affect NVDA?").

## Usage
The graph is designed to be ingested by the `OmniGraphLoader` script (`scripts/load_omni_graph.py`), which constructs a `networkx` graph object at runtime for the MetaOrchestrator or UI backend.

## Roadmap
*   **Expansion:** Continue adding Hero Dossiers for top S&P 500 constituents.
*   **Dynamic Loading:** Implement lazy loading for the Constellation tier to support thousands of nodes.
*   **Live Updates:** Integrate with market data APIs to update Tier 1 nodes in real-time.
