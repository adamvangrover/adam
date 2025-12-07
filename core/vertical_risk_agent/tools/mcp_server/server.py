try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback for environments without MCP installed
    class FastMCP:
        def __init__(self, name): self.name = name
        def resource(self, path): return lambda f: f
        def tool(self): return lambda f: f
        def run(self): print("MCP Server Mock Run (Dependencies missing)")

import sqlite3
import pandas as pd
import sys
import os
import asyncio
import json
from typing import List, Dict, Any, Optional

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import Core Engines
try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
    from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine
except ImportError:
    print("Warning: Could not import Quantum/GenAI engines. Tools will be unavailable.")
    GenerativeRiskEngine = None
    QuantumMonteCarloEngine = None

try:
    from core.engine.meta_orchestrator import MetaOrchestrator
except ImportError:
    print("Warning: MetaOrchestrator import failed. Orchestration tools will fail.")
    MetaOrchestrator = None

# Global Orchestrator Singleton
_orchestrator = None

def get_orchestrator_instance():
    global _orchestrator
    if _orchestrator is None and MetaOrchestrator:
        try:
            _orchestrator = MetaOrchestrator()
        except Exception as e:
            print(f"Failed to initialize MetaOrchestrator: {e}")
    return _orchestrator

# Initialize the MCP Server
mcp = FastMCP("Adam Financial Data Room")

DB_PATH = "finance_data.db"

# --- Resources ---

@mcp.resource("finance://{ticker}/{year}/10k")
def get_10k_filing(ticker: str, year: str) -> str:
    """
    Returns the full text of the 10-K filing for the given ticker and year.
    Resource URI: finance://AAPL/2023/10k
    """
    return f"Full 10-K text content for {ticker} in {year}..."

@mcp.resource("finance://{ticker}/ratios")
def get_financial_ratios(ticker: str) -> str:
    """
    Returns key financial ratios as a CSV string.
    Resource URI: finance://AAPL/ratios
    """
    return "Ratio,Value\nDebt/EBITDA,2.5x\nInterest Coverage,5.0x\nROE,15%"

@mcp.resource("system://repo/assessment")
def get_repo_assessment() -> str:
    """
    Returns the latest repository assessment and plan.
    """
    try:
        with open("docs/REPO_ASSESSMENT_AND_PLAN.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Assessment not found."

# --- Tools ---

@mcp.tool()
def query_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the local financial database.
    """
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_markdown()
    except Exception as e:
        return f"SQL Error: {str(e)}"

@mcp.tool()
def get_covenant_definitions(doc_id: str) -> str:
    """
    Retrieves the legal definitions of financial covenants from a specific credit agreement.
    """
    return """
    Section 7.1. Financial Covenants.
    (a) Consolidated Leverage Ratio. The Borrower shall not permit the Consolidated Leverage Ratio
        as of the end of any Fiscal Quarter to be greater than 4.50 to 1.00.
    """

@mcp.tool()
def simulate_quantum_merton_model(asset_value: float, debt: float, volatility: float, horizon: float) -> str:
    """
    Runs an End-to-End Quantum Monte Carlo simulation for credit risk (Merton Model).
    """
    if not QuantumMonteCarloEngine:
        return "Error: QMC Engine not available."

    qmc = QuantumMonteCarloEngine()
    result = qmc.simulate_merton_model(asset_value, debt, volatility, 0.05, horizon)
    return str(result)

@mcp.tool()
def generate_stress_scenarios(regime: str = "stress", n_samples: int = 5) -> str:
    """
    Generates synthetic market scenarios using a Generative Risk Engine (GAN-based).
    """
    if not GenerativeRiskEngine:
        return "Error: Generative Risk Engine not available."

    engine = GenerativeRiskEngine()
    scenarios = engine.generate_scenarios(n_samples=n_samples, regime=regime)
    return "\n".join([str(s) for s in scenarios])

@mcp.tool()
async def run_deep_dive_analysis(query: str) -> str:
    """
    Triggers a full v23.5 Deep Dive analysis using the MetaOrchestrator.
    Returns the JSON result as a string.
    """
    orchestrator = get_orchestrator_instance()
    if not orchestrator:
        return "Error: MetaOrchestrator not available."

    try:
        result = await orchestrator.route_request(query)
        return str(result)
    except Exception as e:
        return f"Error running Deep Dive: {str(e)}"

@mcp.tool()
def get_snc_rating(borrower_id: str) -> str:
    """
    Retrieves the Shared National Credit (SNC) rating for a borrower.
    Returns a mock rating if not found.
    """
    # In reality this would query a database
    mock_db = {
        "AAPL": "Pass",
        "TSLA": "Pass",
        "AMC": "Substandard",
        "GME": "Special Mention"
    }
    return mock_db.get(borrower_id.upper(), "Not Rated")

@mcp.tool()
def get_esg_score(company: str) -> str:
    """
    Retrieves the ESG score for a company.
    """
    return json.dumps({
        "company": company,
        "environment": 85,
        "social": 78,
        "governance": 90,
        "total": 84
    })

@mcp.tool()
def list_active_agents() -> str:
    """
    Lists all agents currently loaded in the orchestrator.
    """
    orch = get_orchestrator_instance()
    if not orch or not orch.legacy_orchestrator:
        return "Orchestrator not initialized."

    agents = list(orch.legacy_orchestrator.agents.keys())
    return f"Active Agents ({len(agents)}): {', '.join(agents)}"

@mcp.tool()
def get_agent_status(agent_name: str) -> str:
    """
    Checks if a specific agent is loaded.
    """
    orch = get_orchestrator_instance()
    if not orch or not orch.legacy_orchestrator:
        return "Orchestrator not initialized."

    agent = orch.legacy_orchestrator.get_agent(agent_name)
    if agent:
        return f"Agent '{agent_name}' is ACTIVE. Type: {type(agent).__name__}"
    else:
        return f"Agent '{agent_name}' is NOT FOUND."

if __name__ == "__main__":
    # Ensure DB exists
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS financials
                     (ticker text, year int, revenue real, ebitda real)''')
        c.execute("INSERT INTO financials VALUES ('AAPL', 2023, 383285, 114301)")
        conn.commit()
        conn.close()

    mcp.run()
