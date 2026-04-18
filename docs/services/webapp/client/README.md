# React Component Library

The modern frontend presentation layer is housed in `services/webapp/client/`. It is a React application styled with a "Bloomberg Terminal meets Cyberpunk" aesthetic.

## Interactive Workflows

### The 3-Stage Paradigm
Complex data visualization components (e.g., `MarketMayhem.tsx`) generally adhere to a 3-stage interactive workflow:
1. **Directory**: High-level scanning of active entities or distressed assets in a tabular or grid view.
2. **Tearsheet**: A summarized, intermediate view focusing on key metrics of a selected entity.
3. **Drill-Down**: Deep, granular data (often pulling from the Rust pricing layer or graph engine) providing full transparency into the AI's logic.

## State Management Performance
When building React components that subscribe to global states (e.g., Zustand) and render large lists, always extract the inline mapped list or grid items into separate `React.memo()` components. This is crucial to prevent `O(N)` re-renders during high-throughput state updates.
