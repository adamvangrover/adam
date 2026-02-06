# Adam Core Engine

The `core/` directory is the heart of the Adam system, implementing the "System 2" reasoning engine, agents, and specialized financial modules.

---

## 🧠 The Reasoning Engine (`core/engine/`)

The engine moves beyond simple "Chain-of-Thought" prompting to a **Cyclical Reasoning Architecture** implemented with `LangGraph`.

### 1. Neuro-Symbolic Planner
*   **File:** `core/engine/neuro_symbolic_planner.py`
*   **Function:** Decomposes a high-level user query (e.g., "Analyze AAPL") into a Directed Acyclic Graph (DAG) of executable tasks.
*   **Logic:** Uses an LLM to generate the plan, but validates the graph structure with symbolic logic (NetworkX) to ensure no cycles or impossible paths.

### 2. Meta Orchestrator
*   **File:** `core/engine/meta_orchestrator.py`
*   **Function:** The central router. It receives the user input and decides:
    *   **Fast Path (System 1):** Route to Swarm for news/data.
    *   **Slow Path (System 2):** Route to Planner for deep analysis.
*   **Pattern:** Uses semantic routing (embedding similarity) to classify intent.

### 3. Consensus Engine
*   **File:** `core/engine/consensus_engine.py`
*   **Function:** Aggregates outputs from multiple agents.
*   **Algorithm:**
    *   If agents agree (Variance < Threshold) -> **Pass**.
    *   If agents disagree -> **Trigger Debate** (Adversarial Loop).

---

## 🛡️ The Credit Sentinel (`core/credit_sentinel/`)

The **Credit Sentinel** is Adam's flagship module for Distressed Debt and Credit Risk analysis.

### Architecture
It functions as a "Sub-Graph" within the main system.

1.  **Ingestion:** Fetches 10-K/10-Q (XBRL) and Earnings Transcripts.
2.  **Ratio Calculator (`agents/ratio_calculator.py`):** Deterministic Python functions for EBITDA, Net Debt, Interest Coverage.
3.  **Risk Analyst (`agents/risk_analyst.py`):** An LLM agent that reads the MD&A section to identify qualitative risks (Litigation, Management Tone).
4.  **Synthesizer:** Merges Quantitative and Qualitative data into a **SNC Rating** (Shared National Credit).

### Key Files
*   `models/distress_classifier.py`: Random Forest model for predicting bankruptcy probability.
*   `data_ingestion/financial_statements.py`: Handlers for SEC EDGAR API.

---

## 📂 Directory Structure

| Directory | Description |
| :--- | :--- |
| `agents/` | General-purpose agents (Fundamental, Technical, Sentinel). |
| `credit_sentinel/` | Specialized module for credit risk. |
| `engine/` | The core reasoning logic (Planner, Orchestrator, Graph). |
| `data_processing/` | The "Universal Ingestor" pipeline. |
| `system/` | System-level utilities and Async Swarm infrastructure. |

---

## 🚀 Developer Guide

To run the core engine in isolation (without the webapp):

```bash
# Run the Interactive CLI
python scripts/run_adam.py

# Run a specific agent test
python scripts/run_adam.py --agent "RiskAnalyst" --query "Assess liquidity for TSLA"
```

**Note:** Always refer to `AGENTS.md` in the root directory for coding standards.
