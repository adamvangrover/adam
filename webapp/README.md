# Adam v26.0 Neural Dashboard

The **Neural Dashboard** (Protocol ADAM-V-NEXT) is the primary command center for the Adam system. It provides real-time visualization of agent thoughts, risk metrics, and simulation states.

## 🚀 Launching the Dashboard

The dashboard is a static frontend that connects to the Adam API.

### Prerequisites
*   Node.js & npm/pnpm
*   Running Adam Backend (see `services/webapp/README.md`)

### Quick Start

```bash
cd webapp
pnpm install
pnpm dev
```

## 🖥️ Key Interfaces

### 1. The Synthesizer (`/synthesizer`)
Aggregates signals from all active agents into a single "Confidence Score" gauge. It visualizes the consensus mechanism in real-time.

### 2. Command Console (`/console`)
A terminal-like interface for direct interaction with the `MetaOrchestrator`. Supports natural language queries and displays rich outputs (markdown, tables).

### 3. Crisis Simulator (`/simulator`)
A dedicated view for running and visualizing macro-economic stress tests (e.g., Liquidity Shock, Sovereign Default).

## 🛠️ Development

For frontend development instructions (Vite, React, TypeScript), please refer to [README_DEV.md](README_DEV.md).
