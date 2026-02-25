# Credit Memo Pipeline Documentation

## Overview
This pipeline orchestrates the generation of institutional-grade credit memos using a multi-agent neuro-symbolic architecture. It integrates financial modeling (ICAT), quantitative risk assessment, and legal document review into a unified workflow.

## Architecture

### 1. Agents
*   **RiskAssessmentAgent (`core/agents/risk_assessment_agent.py`):** Calculates PD, LGD, and Risk-Weighted Assets (RWA) using Merton distance models and Altman Z-Score integration.
*   **LegalAgent (`core/agents/legal_agent.py`):** Scans credit agreements for key covenants (Negative Pledge, Cross-Default) and 10-K filings for fraud signals.
*   **CreditMemoOrchestrator (`core/agents/orchestrators/credit_memo_orchestrator.py`):** The central nervous system that coordinates data flow, simulates agent "interlock" (negotiation/verification), and generates the final JSON artifacts.

### 2. Engines
*   **ICAT Engine (`core/engine/icat.py`):** "Ingest, Clean, Analyze, Transform". Handles LBO modeling, DCF valuation, and ratio spreading.

## How to Run

### Generate Memos
To run the full pipeline and regenerate the static data for the frontend:

```bash
python scripts/generate_credit_memos.py
```

This will populate `showcase/data/` with:
*   `credit_memo_*.json`: Individual credit reports.
*   `credit_memo_library.json`: Index of all reports.
*   `risk_legal_interaction.json`: Logs of the agent interlock for frontend visualization.

### Run Tests
```bash
export PYTHONPATH=$PYTHONPATH:.
python tests/test_credit_orchestrator.py
```

## Features
*   **Simulated RAG:** The orchestrator extracts relevant chunks from mock documents to generate realistic citations.
*   **Agent Interlock:** Simulates a live dialogue between Risk and Legal agents, generating "UI Events" that the frontend uses to choreograph cursor movements and focus.
*   **Graceful Fallback:** The system is designed to work with mock data if external data sources (EDGAR, Yahoo Finance) are unavailable.
