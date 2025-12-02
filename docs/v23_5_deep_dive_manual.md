# Adam v23.5 "AI Partner" - Deep Dive Protocol Manual

## Overview
The v23.5 "AI Partner" upgrade transforms Adam from a simple research assistant into a full-spectrum **Autonomous Financial Analyst**. It introduces a 5-Phase "Deep Dive" execution protocol designed to mimic the workflow of a senior institutional investor.

## The 5-Phase Protocol

### Phase 1: Entity, Ecosystem & Management (The Foundation)
*   **Goal**: Establish a foundational understanding of the business quality.
*   **Key Analyses**:
    *   **Entity Resolution**: Legal hierarchy and jurisdiction.
    *   **Management Assessment**: Capital allocation track record and alignment.
    *   **Competitive Positioning**: Moat analysis (Wide/Narrow/None) and Technology Risk.

### Phase 2: Deep Fundamental & Valuation (The Equity Lens)
*   **Goal**: Determine the intrinsic value of the equity.
*   **Key Analyses**:
    *   **Fundamentals**: Revenue CAGR, EBITDA margin trends.
    *   **DCF Model**: WACC, Terminal Growth, Intrinsic Share Price.
    *   **Multiples Analysis**: EV/EBITDA vs Peers.
    *   **Price Targets**: Bear, Base, and Bull cases.

### Phase 3: Credit, Covenants & SNC Ratings (The Debt Lens)
*   **Goal**: Assess the creditworthiness and downside protection.
*   **Key Analyses**:
    *   **SNC Rating**: Regulatory rating (Pass, Special Mention, Substandard) for each facility.
    *   **Covenant Analysis**: Headroom against primary constraints.
    *   **Structure**: Collateral coverage and priority of claims.

### Phase 4: Risk, Simulation & Quantum Modeling (The Stress Test)
*   **Goal**: Stress test the investment thesis against tail risks.
*   **Key Analyses**:
    *   **Monte Carlo**: Default probability estimation.
    *   **Quantum Scenarios**: Impact of low-probability, high-impact events (e.g., Geopolitical shocks).
    *   **Trading Dynamics**: Short interest and liquidity risk.

### Phase 5: Synthesis, Conviction & Strategy (The Verdict)
*   **Goal**: Formulate a final actionable recommendation.
*   **Key Analyses**:
    *   **M&A Posture**: Is the company a Buyer or a Target?
    *   **Final Verdict**: Buy/Sell/Hold.
    *   **Conviction Level**: 1-10 score.
    *   **Rationale**: Explicit reasoning trace.

## Architecture

The system uses a **Hyper-Dimensional Knowledge Graph (HDKG)** as its state object, defined in `core/schemas/v23_5_schema.py`. This state is populated sequentially by the `DeepDiveGraph` (`core/engine/deep_dive_graph.py`).

### Routing
The `MetaOrchestrator` detects high-complexity intents (e.g., "Deep Dive on Apple", "Full Analysis") and routes them to the `DeepDiveGraph` instead of the legacy `AgentOrchestrator`.

## Usage
To trigger the Deep Dive protocol, simply ask the system:
> "Run a full deep dive analysis on AAPL."
> "Act as my AI Partner and analyze Tesla's valuation and credit risk."
