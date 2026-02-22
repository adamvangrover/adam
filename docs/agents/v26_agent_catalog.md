# Adam v26.0 Agent Catalog

A comprehensive reference for the core agents in the system.

## Meta-Agents

### 1. Meta-Orchestrator (`core/engine/meta_orchestrator.py`)
*   **Role:** The Central Nervous System.
*   **Function:** Routes queries to Swarm (System 1) or Graph (System 2).
*   **Key Method:** `route_request(query, context)`

### 2. Neuro-Symbolic Planner (`core/engine/neuro_symbolic_planner.py`)
*   **Role:** The Architect.
*   **Function:** Decomposes complex goals into a DAG of tasks.
*   **Output:** `ExecutionPlan` object.

## Specialized Agents (System 2)

### 3. FundamentalAnalyst (`core/agents/fundamental_analyst_agent.py`)
*   **Role:** Equity Research.
*   **Inputs:** Ticker, Financial Statements.
*   **Outputs:** DCF Valuation, Key Ratios.

### 4. RiskAssessmentAgent (`core/agents/risk_assessment_agent.py`)
*   **Role:** Credit Officer.
*   **Inputs:** Balance Sheet, Debt Schedule.
*   **Outputs:** Default Probability, Recovery Rate.

### 5. ManagementAssessmentAgent
*   **Role:** Qualitative Auditor.
*   **Inputs:** Earnings Call Transcripts, Insider Filings.
*   **Outputs:** Management Score (0-10), Alignment Analysis.

## Swarm Workers (System 1)

### 6. NewsScanner
*   **Role:** Perception.
*   **Function:** Continuous polling of RSS/API feeds.
*   **Action:** Emits `NEWS_EVENT` to Message Bus.

### 7. SentinelWorker
*   **Role:** Anomaly Detection.
*   **Function:** Monitors numerical data streams for outliers (e.g., volume spikes).
