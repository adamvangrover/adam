# ADAM OS — Agent Network Registry

## 1. Underwriting Agent
*   **Role:** Analyzes credit metrics, financial statements, and covenant compliance.
*   **Bounded Context:** `Credit Underwriting`
*   **State Schema:** `UnderwritingState` (Includes `ebitda_margin`, `leverage_ratio`, `fccr`).
*   **Required Tools:** `extract_financials`, `evaluate_covenant_jsonlogic`.
*   **JIT Memory Strategy:** Semantic search over trailing 12-month (TTM) SEC filings via Qdrant.

## 2. Surveillance Agent
*   **Role:** Continuous monitoring of portfolio companies for distress signals.
*   **Bounded Context:** `Portfolio Surveillance`
*   **State Schema:** `SurveillanceState` (Includes `news_sentiment_score`, `liquidity_runway_days`).
*   **Required Tools:** `fetch_market_data`, `trigger_temporal_alert`.
*   **JIT Memory Strategy:** Episodic memory retrieval of prior quarter earnings call transcripts.

## 3. Orchestrator (System Architect)
*   **Role:** Routes tasks, manages context windows, and enforces PROV-O telemetry.
*   **Bounded Context:** `Workflow Runtime`
*   **State Schema:** `OrchestrationState` (Includes `active_agents`, `trace_id`, `execution_graph`).
*   **Required Tools:** `delegate_task`, `checkpoint_state`.
