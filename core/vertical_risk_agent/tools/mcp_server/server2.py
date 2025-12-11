try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback for environments without MCP installed
    class FastMCP:
        def __init__(self, name): self.name = name
        def resource(self, path): return lambda f: f
        def tool(self): return lambda f: f
        def run(self): print("MCP Server Mock Run")

import sys
import os
import asyncio
from typing import List

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from core.vertical_risk_agent.tools.agent_tools import AgentTools, FinancialRatio

# Initialize Tools
agent_tools = AgentTools()

# Initialize the MCP Server
mcp = FastMCP("Financial Data Room")

@mcp.resource("finance://{ticker}/{year}/10k")
def get_10k_filing(ticker: str, year: str) -> str:
    """
    Returns the full text of the 10-K filing for the given ticker and year.
    Resource URI: finance://AAPL/2023/10k
    """
    return agent_tools.get_10k_filing(ticker, year)

@mcp.resource("finance://{ticker}/ratios")
def get_financial_ratios(ticker: str) -> str:
    """
    Returns key financial ratios as a string representation.
    Resource URI: finance://AAPL/ratios
    """
    ratios = agent_tools.get_financial_ratios(ticker)
    # Convert list of models to string for simple MCP consumption
    return "\n".join([f"{r.name},{r.value}" for r in ratios])

@mcp.tool()
def query_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the local financial database.
    Use this to aggregate data or find specific numerical facts.
    """
    return agent_tools.query_sql(query)

@mcp.tool()
def get_covenant_definitions(doc_id: str) -> str:
    """
    Retrieves the legal definitions of financial covenants from a specific credit agreement.
    """
    return agent_tools.get_covenant_definitions(doc_id)

@mcp.tool()
def simulate_quantum_merton_model(asset_value: float, debt: float, volatility: float, horizon: float) -> str:
    """
    Runs an End-to-End Quantum Monte Carlo simulation for credit risk (Merton Model).
    Returns the Probability of Default (PD) and Asset Value stats.
    """
    result = agent_tools.simulate_quantum_merton_model(asset_value, debt, volatility, horizon)
    return str(result)

@mcp.tool()
def generate_stress_scenarios(regime: str = "stress", n_samples: int = 5) -> str:
    """
    Generates synthetic market scenarios using a Generative Risk Engine (GAN-based).
    Useful for tail risk analysis. Regime can be 'normal', 'stress', or 'crash'.
    """
    result = agent_tools.generate_stress_scenarios(regime, n_samples)
    if isinstance(result, list):
        return "\n".join([str(s) for s in result])
    return str(result)

@mcp.tool()
async def run_deep_dive_analysis(query: str) -> str:
    """
    Triggers a full v23.5 Deep Dive analysis using the MetaOrchestrator.
    Returns the JSON result as a string.
    """
    return await agent_tools.run_deep_dive_analysis(query)

if __name__ == "__main__":
    mcp.run()
