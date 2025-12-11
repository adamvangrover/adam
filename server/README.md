# Adam Financial Engine (MCP Server)

This directory contains the **Model Context Protocol (MCP)** server for Adam. It exposes the core financial intelligence capabilities as executable tools.

## 🛠️ Exposed Capabilities

### 1. Quantum & Risk
*   `run_quantum_simulation(asset, debt, vol)`: Uses `core/v22_quantum_pipeline` to simulate credit risk using Quantum Monte Carlo.
*   `generate_market_scenarios(regime)`: Uses `core/vertical_risk_agent` (GAN) to generate stress scenarios.

### 2. Credit & SNC (Shared National Credit)
*   `analyze_snc_credit(financials, capital_structure)`: Deploys `SNCRatingAgent` to classify debt facilities (Pass, Special Mention, Substandard).
*   `analyze_covenants(leverage, threshold)`: Deploys `CovenantAnalystAgent` to assess breach risk.

### 3. Strategy & Planning
*   `plan_workflow(start, target)`: Uses `NeuroSymbolicPlanner` to discover reasoning paths in the Knowledge Graph.
*   `compare_peers(company_id)`: Deploys `PeerComparisonAgent` for relative valuation.

### 4. Data Operations
*   `ingest_file(path)`: Uses `UniversalIngestor` to scrub and index documents into the Gold Standard artifacts.
*   `retrieve_market_data(ticker)`: Fetches real-time market data.

### 5. Utilities
*   `execute_python_sandbox(code)`: Secure execution environment for ad-hoc analysis.

## 📚 Resources
*   `adam://project/manifest`: Capabilities summary.
*   `adam://docs/{filename}`: Read any system documentation dynamically.

## 🚀 Running the Server

### Standalone
```bash
python server/mcp_server.py
```

### via Claude Desktop / IDE
Use the root `mcp.json` to configure your client.
