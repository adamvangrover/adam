# FO Super-App System Architecture

## 1. System Overview

The **FO Super-App** is a unified Front Office application designed to integrate markets, pricing, rating, execution, analytics, and autonomous strategy control. It acts as a "co-pilot brain" for financial professionals, merging Investment Banking (IB), Wealth Management (WM), and Asset Management (AM) capabilities.

The platform is designed to:
1.  **Price competitively** against institutional desks.
2.  **Generate and execute strategies** using real-time and historical data.
3.  **Unify perspectives** across IB, WM, and AM.
4.  **Provide a personal sandbox** for secure, local knowledge retention.
5.  **Support multi-agent control** via the Model Context Protocol (MCP).

## 2. Core Architecture

The system is structured into five primary layers:

### 2.1 Market & Execution Layer
*   **Pricing Engine:** Institutional-grade multi-asset pricing.
*   **Market Data:** Real-time feeds and historical replay.
*   **Execution Router:** Benchmark-aware order routing and simulation.

### 2.2 Strategy & Analysis Layer
*   **Strategy Engine:** Alpha signal ingestion, LLM-assisted drafting, and RL optimization.
*   **Cross-Asset Engine:** Hedging and derivatives logic.
*   **Scenario Engine:** "Market Mayhem" stress testing.

### 2.3 Credit & Risk Layer
*   **Credit Ratings:** PD scoring (S&P-like), regulatory rating assignment, and credit memo generation.
*   **Risk Engine:** Real-time VAR, Greeks, convexity, and cross-exposure metrics.

### 2.4 Personal Memory Layer
*   **Local Vector Store:** SQLite + Embeddings (Chroma/FAISS) for durable long-term memory.
*   **Knowledge Graph:** Stores user preferences, investment philosophy, and research trails.
*   **Privacy:** Secure offline retention with user-controlled permissions.

### 2.5 Control Layer (MCP)
*   **LLM Integration:** A plugin-based architecture using the Model Context Protocol (MCP).
*   **Agent Stack:** Specialized agents for Strategy, Risk, Market, Execution, Memory, and Compliance.
*   **Tooling:** Standardized `invoke`/`validate`/`schema` interfaces for all modules.

## 3. Directory Structure

The repository is organized to support this modular architecture:

```
/core/
 ├── pricing_engine/       # Market making and asset pricing
 ├── credit_ratings/       # Credit scoring and regulatory ratings
 ├── market_data/          # Data feeds and normalization
 ├── execution_router/     # Order management and routing
 ├── risk_engine/          # Risk metrics and simulation
 ├── strategy/             # Alpha signals and RL optimization
 ├── mcp/                  # Model Context Protocol tools and agents
 ├── memory/               # Personal Knowledge Graph engine
```

## 4. System Modes

*   **Console Mode ("Trader Terminal"):** Command-line interface for power users and scripting.
*   **Guided Mode ("Wealth + Family Office"):** Natural language interface for recommendations and portfolio actions.
*   **Developer Mode:** Direct API access and tool inspection.

## 5. Technology Stack

*   **Backend:** Python (Core logic, Agents, Analytics)
*   **Frontend:** React (Web UI), Terminal (CLI)
*   **Data:** SQLite, Redis, Vector Stores (Chroma/FAISS)
*   **AI/ML:** PyTorch, LLM integration via MCP
