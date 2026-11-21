# Adam Platform v24.0: Architectural Blueprint

## 1. Executive Strategy
The **Adam Platform v24.0** represents a fundamental re-platforming from a "System of Record" to an autopoietic "System of Agency". It addresses the "Great Divergence" in financial markets by unifying Investment Banking (IB), Wealth Management (WM), and Asset Management (AM) into a **Unified Financial Operating System (UFOS)**.

## 2. Architecture: Polyglot Core
The system utilizes a high-discipline Monorepo structure:
- **Iron Core (Rust)**: Handles high-frequency execution, order matching, and risk checks. Located in `/iron_core`.
- **Intelligence Layer (Python)**: Orchestrates LLMs and agentic workflows. Located in `/intelligence_layer`.
- **Interface (TypeScript/React)**: A "thick client" cockpit. Located in `/interface`.

## 3. The Unified Ledger
Data is persisted via a **Dual-Storage Strategy**:
- **Hot/Warm Store**: Time-series data (TimescaleDB/Redis) for the Order Book.
- **Cold Store**: Vector Database and Temporal Knowledge Graph for semantic context.

## 4. Key Components

### HNASP (Hybrid Neurosymbolic Agent State Protocol)
Governs agent behavior using:
- **JsonLogic**: Deterministic rule enforcement (Constitution).
- **BayesACT**: Probabilistic persona maintenance (EPA vectors).

### Model Context Protocol (MCP)
Acts as the universal bus connecting the Iron Core tools to the Python agents.
- **Resources**: Real-time market data streams.
- **Tools**: Executable functions (e.g., `execute_trade`) with Human-in-the-Loop protection.

### Meta-Agents
1.  **Evolutionary Architect**: Autonomous DevOps agent that refactors code using AST mutation and "The Gauntlet" verification.
2.  **Chronos**: Manages Temporal Memory (HTM) and the Bitemporal Knowledge Graph (Valid Time vs. Transaction Time).
3.  **Didactic Architect**: Ensures interpretability by detecting documentation drift and generating interactive Marimo tutorials.

## 5. Directory Structure
```
adam_v24/
├── iron_core/          # Rust Execution Engine
├── intelligence_layer/ # Python Agents & MCP Server
│   ├── hnasp/          # Governance & Persona
│   └── architects/     # Meta-Agents (Evolutionary, Chronos, Didactic)
├── interface/          # React/TypeScript Frontend
└── schemas/            # SQL & Pydantic Definitions
```

This architecture ensures that Adam v24.0 is not just a tool, but a sovereign, self-evolving financial infrastructure.
