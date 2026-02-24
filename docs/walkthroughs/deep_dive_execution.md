# Walkthrough: The Deep Dive Execution Logic

The "Deep Dive" is the flagship capability of Adam v26.0. It is a fully autonomous pipeline that produces an institutional-grade investment memo.

## The 5-Phase Protocol

### Phase 1: Entity Resolution (`core/engine/deep_dive_graph.py`)
*   **Goal:** Establish identity and context.
*   **Action:**
    1.  Resolve Ticker/Name to LEI (Legal Entity Identifier).
    2.  Map corporate hierarchy (Subsidiaries).
    3.  Assess Management (Insider buying, tenure, track record).
    4.  Determine "Moat" status (Wide/Narrow/None).

### Phase 2: Fundamentals & Valuation
*   **Goal:** Calculate intrinsic value.
*   **Action:**
    1.  **DCF:** 10-year projection with terminal value.
    2.  **Multiples:** Compare EV/EBITDA against peer median.
    3.  **Trend:** Analyze CAGR of Revenue and Margins.

### Phase 3: Credit & Insolvency
*   **Goal:** Downside protection.
*   **Action:**
    1.  **SNC Rating:** Assign a regulatory grade (Pass/Substandard).
    2.  **Covenant Analysis:** Find the "Choke Point" (tightest covenant).
    3.  **Liquidity:** Calculate months of runway.

### Phase 4: Stochastic Risk
*   **Goal:** Stress testing.
*   **Action:**
    1.  **Monte Carlo:** 10,000 runs of EBITDA volatility.
    2.  **Quantum Scenarios:** Simulate "Black Swan" events (e.g., War, Pandemic).

### Phase 5: Synthesis
*   **Goal:** The Verdict.
*   **Action:**
    1.  Weigh Equity Upside vs. Credit Downside.
    2.  Assign Conviction Score (1-10).
    3.  Generate Natural Language Memo.

## Data Flow
Data flows strictly from Phase 1 -> 5. Each phase appends to the `v26_knowledge_graph` state object.

*   Phase 4 reads `EBITDA` generated in Phase 2.
*   Phase 5 reads `Default Probability` generated in Phase 4.

## Error Handling
If any node fails (e.g., Data API down), the graph can:
1.  **Retry:** Standard exponential backoff.
2.  **Fallback:** Switch to a heuristic estimation mode (flagged as "Low Confidence").
