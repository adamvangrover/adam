# Adam Agent Guide

This guide details the specialized autonomous agents within the Adam v26.0 architecture.

## Overview

Adam utilizes a **Hybrid Cognitive Engine** (System 1 + System 2) where agents operate as specialized nodes in a neuro-symbolic graph.

- **System 1 (Swarm):** Fast, asynchronous agents for perception and monitoring.
- **System 2 (Graph):** Deliberative, synchronous agents for reasoning and analysis.

## Core Agents

### 1. Fundamental Analyst Agent
*   **Path:** `core/agents/fundamental_analyst_agent.py`
*   **Role:** Performs deep dive financial analysis on specific tickers.
*   **Capabilities:**
    *   **Ratio Analysis:** Calculates profitability, liquidity, and solvency ratios.
    *   **Valuation:** Performs DCF (Discounted Cash Flow) analysis.
    *   **Health Assessment:** Generates a financial health score (Strong/Moderate/Weak).
    *   **Report Generation:** Synthesizes findings into a structured summary.
*   **Inputs:** `AgentInput` with a ticker query (e.g., "AAPL").
*   **Outputs:** `AgentOutput` containing the analysis summary, confidence score, and structured metadata.

### 2. Data Retrieval Agent
*   **Role:** Fetches raw financial data from external sources (APIs, databases).
*   **Interaction:** Called by analytical agents via the Swarm protocol.

### 3. Technical Analyst Agent
*   **Role:** Analyzes price action, trends, and technical indicators.

## Developing New Agents

Refer to `AGENTS.md` for strict development standards.

1.  **Inherit:** Extend `AgentBase`.
2.  **Schema:** Use `AgentInput` and `AgentOutput`.
3.  **Tests:** Implement unit tests in `tests/`.
4.  **Register:** Add to the Meta Orchestrator.

## Testing

Run agent tests using `pytest`:

```bash
pytest tests/test_fundamental_analyst.py
```
