try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback for environments without MCP installed
    class FastMCP:
        def __init__(self, name): self.name = name
        def resource(self, path): return lambda f: f
        def tool(self): return lambda f: f
        def run(self): print("MCP Server Mock Run")

import sqlite3
import pandas as pd
import sys
import os
from typing import List

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
    from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine
except ImportError:
    print("Warning: Could not import Quantum/GenAI engines. Tools will be unavailable.")
    GenerativeRiskEngine = None
    QuantumMonteCarloEngine = None

try:
    from core.engine.meta_orchestrator import MetaOrchestrator
    import asyncio
except ImportError:
    MetaOrchestrator = None

# Global Orchestrator Singleton
_orchestrator = None

def get_orchestrator_instance():
    global _orchestrator
    if _orchestrator is None and MetaOrchestrator:
        _orchestrator = MetaOrchestrator()
    return _orchestrator

# Initialize the MCP Server
mcp = FastMCP("Financial Data Room")

DB_PATH = "finance_data.db"

@mcp.resource("finance://{ticker}/{year}/10k")
def get_10k_filing(ticker: str, year: str) -> str:
    """
    Returns the full text of the 10-K filing for the given ticker and year.
    Resource URI: finance://AAPL/2023/10k
    """
    # In a real app, this would read from the file system or S3
    return f"Full 10-K text content for {ticker} in {year}..."

@mcp.resource("finance://{ticker}/ratios")
def get_financial_ratios(ticker: str) -> str:
    """
    Returns key financial ratios as a CSV string.
    Resource URI: finance://AAPL/ratios
    """
    return "Ratio,Value\nDebt/EBITDA,2.5x\nInterest Coverage,5.0x"

@mcp.tool()
def query_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the local financial database.
    Use this to aggregate data or find specific numerical facts.
    """
    # Security check: only allow SELECT
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
    # logic to parse the credit agreement text
    return """
    Section 7.1. Financial Covenants.
    (a) Consolidated Leverage Ratio. The Borrower shall not permit the Consolidated Leverage Ratio
        as of the end of any Fiscal Quarter to be greater than 4.50 to 1.00.
    """

@mcp.tool()
def simulate_quantum_merton_model(asset_value: float, debt: float, volatility: float, horizon: float) -> str:
    """
    Runs an End-to-End Quantum Monte Carlo simulation for credit risk (Merton Model).
    Returns the Probability of Default (PD) and Asset Value stats.
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
    Useful for tail risk analysis. Regime can be 'normal', 'stress', or 'crash'.
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

if __name__ == "__main__":
    mcp.run()
