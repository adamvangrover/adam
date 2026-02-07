Swarm Manager Operations Manual & Tutorial

Overview

This guide provides a comprehensive walkthrough for setting up, operating, and extending the Adam Swarm Architecture. It covers runtime setups, UI deployment, and managing the "Financial Digital Twin" simulations.

1. Runtime Setup

Prerequisites

Docker & Docker Compose

Python 3.10+

Node.js 18+ (for UI)

Rust (cargo) (for high-performance engines)

Initialization

Configure Environment:
Ensure .env is populated with valid API keys (OpenAI, Gemini, Neo4j credentials).

cp .env.example .env


Boot the Infrastructure:
Launch the database and broker layers.

docker-compose up -d neo4j qdrant redis


Start the Swarm Manager:
This initializes the orchestrator defined in core/engine/swarm/swarm_manager.py.

python scripts/boot_system.py --mode=swarm --config=config/swarm_runtime_setup.yaml


2. Microservices & API Integration

The Swarm relies on a "portable and modular" microservice architecture.

MCP Server (Model Context Protocol):
Located at core/v30_architecture/python_intelligence/mcp/server.py. This standardizes tool usage for LLMs.

To Extend: Add new tools to core/mcp/tools.json.

Rust Pricing Engine:
Located in core/v30_architecture/rust_core/.

To Build: cd core/v30_architecture/rust_core && cargo build --release

Usage: Accessed via gRPC by the Python QuantitativeAgent.

3. Building Knowledge Graphs

The Swarm Manager uses the FinancialTwinBuilder agent to maintain the graph.

Step-by-Step Graph Build:

Ingest Data:

python scripts/run_daily_ingestion.py --source=sec_filings --tickers=NVDA,AMD


Vectorize & Graph:
The SwarmManager automatically triggers embedding.

Embeddings go to Qdrant.

Relationships (Supplier, Competitor) go to Neo4j based on artifacts/ontologies/adam_credit_risk.ttl.

4. Running Simulations (Market Mayhem)

To run a "Market Mayhem" scenario (e.g., a financial crisis simulation):

Define Scenario:
Create a YAML file in artifacts/simulation/scenarios/ (e.g., hyperinflation_2026.yaml).

Trigger Simulation:
Use the CLI or the Dashboard.

python scripts/run_simulations.sh --scenario=hyperinflation_2026 --agents=50


Analyze Results:
Results are logged to data/simulated_JSONL_output/. The Swarm Manager aggregates these into a "Financial Twin" state update.

5. User Interfaces & Dashboards

The UI is built with React and Tailwind. The Swarm Manager communicates via WebSockets/API.

Launch UI:

cd services/webapp_v24
npm install
npm run dev


Customizing the Dashboard:
The Swarm Manager can generate new UI components on the fly.

Prompt: "Generate a React component to visualize correlation between Oil Prices and Semiconductor Stocks."

The system will output code to services/webapp_v24/components/generated/ and register it in the layout.

6. Math & Logic Engine Usage

For complex calculations, the Swarm delegates to the Logic Engine.

Input: Structured JSON containing financial variables.

Processing: Python numpy/pandas or Rust binaries.

Output: Verified numerical results returned to the context window.

7. Deep Reporting & Documentation

The Swarm Manager enforces a "Deep and Broad" reporting standard.

Logs: All decisions are logged in core/data/decision_log.jsonl with provenance.

Whitepapers: Generated reports (like showcase/report_2026_strategic_outlook.html) include citations and logic traces.

Future Expansion

To add a new capability (e.g., a "Geopolitical Risk" module):

Define the Agent Prompt in prompt_library/swarm/.

Register the Agent in config/swarm_runtime_setup.yaml.

Add necessary Data Tools in core/tools/.

Restart the Swarm.
