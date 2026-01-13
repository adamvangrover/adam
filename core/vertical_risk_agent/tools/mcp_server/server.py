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
import importlib
from typing import List, Dict, Any, Optional

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import Dynamic Registry
try:
    from core.vertical_risk_agent.tools.mcp_server.components import get_component_module
except ImportError:
    print("Warning: Could not import component registry.")
    def get_component_module(name): return None

# Import Core Engines (Fallback if registry fails for core components)
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


@mcp.resource("financial://market/book/{symbol}")
def get_order_book(symbol: str) -> str:
    """
    Returns the current Level 2 order book for a symbol.
    Resource URI: financial://market/book/AAPL
    """
    # Mock order book data for UFOS demo
    book = {
        "symbol": symbol,
        "timestamp": 1698765432,
        "bids": [{"price": 150.23, "size": 500}, {"price": 150.22, "size": 1200}],
        "asks": [{"price": 150.26, "size": 400}, {"price": 150.27, "size": 800}]
    }
    return json.dumps(book, indent=2)


@mcp.resource("financial://portfolio/{id}/risk")
def get_portfolio_risk(id: str) -> str:
    """
    Returns real-time risk metrics (VaR, Beta).
    Resource URI: financial://portfolio/123/risk
    """
    return json.dumps({
        "portfolio_id": id,
        "VaR_95": "1.4%",
        "Beta": 1.2,
        "Sharpe": 2.1,
        "Drawdown": "0.5%"
    }, indent=2)


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


@mcp.resource("adam://knowledge/artifacts")
def list_gold_standard_artifacts() -> str:
    """
    Lists all available Gold Standard data artifacts in the repository.
    Resource URI: adam://knowledge/artifacts
    """
    import os
    gold_std_path = "data/gold_standard/"
    if not os.path.exists(gold_std_path):
        return "Gold Standard directory not found."

    files = [f for f in os.listdir(gold_std_path) if f.endswith('.json') or f.endswith('.jsonl')]
    return json.dumps({"artifacts": files}, indent=2)

# --- Tools ---


@mcp.tool()
def execute_market_order(symbol: str, quantity: int, side: str) -> str:
    """
    Executes a market order. Requires Human-in-the-Loop confirmation.
    """
    # In a real system, this would:
    # 1. Validate against risk limits (Pre-Trade Check)
    # 2. Pop a confirmation dialog to the user via the UI
    # 3. Send to OEMS
    return json.dumps({
        "status": "SENT_TO_CONFIRMATION",
        "order": {"symbol": symbol, "quantity": quantity, "side": side},
        "message": "Please confirm execution in the UI."
    })


@mcp.tool()
def run_backtest(strategy_id: str, start_date: str, end_date: str) -> str:
    """
    Runs a backtest for a given strategy ID over a time range.
    """
    # Simulates spinning up a sandbox container
    return json.dumps({
        "status": "COMPLETED",
        "strategy": strategy_id,
        "pnl_pct": 12.5,
        "sharpe": 1.8,
        "max_drawdown": 0.15,
        "report_url": f"/reports/backtest_{strategy_id}_{start_date}.html"
    })


@mcp.tool()
def query_memory(query: str) -> str:
    """
    Queries the 'Personal Memory' (Vector Store + Knowledge Graph) for qualitative insights.
    """
    # Mock RAG response
    return f"Based on your personal memory: You have a preference for 'Low Volatility' stocks and a restriction against 'Tobacco'. Regarding '{query}': We found 3 related emails from 2023 discussing this topic."


@mcp.tool()
def rebalance_portfolio(portfolio_id: str, target_allocation: str) -> str:
    """
    Calculates and proposes a rebalancing plan.
    target_allocation should be a JSON string like '{"AAPL": 0.5, "GOOG": 0.5}'
    """
    return json.dumps({
        "plan_id": "PLAN-999",
        "trades": [
            {"action": "SELL", "symbol": "MSFT", "quantity": 50},
            {"action": "BUY", "symbol": "AAPL", "quantity": 20}
        ],
        "estimated_commission": 10.00,
        "action": "CONFIRM_REQUIRED"
    })


@mcp.tool()
def query_sql(query: str) -> str:
    """
    Executes a read-only SQL query against the local financial database.
    Restricted to specific tables for security.
    """
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    # Whitelist of allowed tables to prevent accessing secrets or system tables
    ALLOWED_TABLES = {'financials', 'sqlite_sequence'}

    def authorizer(action, arg1, arg2, dbname, source):
        # Allow SELECT statements
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        # Allow READ on whitelisted tables
        if action == sqlite3.SQLITE_READ:
            if arg1 in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        # Allow standard SQL functions
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK
        # Deny everything else (INSERT, UPDATE, DELETE, PRAGMA, etc.)
        return sqlite3.SQLITE_DENY

    try:
        # Use read-only URI mode as an extra layer of defense
        uri_path = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True)
        conn.set_authorizer(authorizer)

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


# --- New Features: Industry Router, Developer Swarm, Autonomous Improvement ---

def _load_module_dynamically(component_key: str) -> Optional[Any]:
    module_path = get_component_module(component_key)
    if not module_path:
        return None
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        print(f"Failed to load component {component_key}: {e}")
        return None

@mcp.tool()
async def analyze_industry(sector: str) -> str:
    """
    Routes analysis request to the specialized agent for the given sector.
    """
    # Map 'tech' to 'technology', etc.
    sector_key = sector.lower()
    if "tech" in sector_key: sector_key = "technology"
    elif "finance" in sector_key: sector_key = "financials"
    elif "energy" in sector_key: sector_key = "energy"

    module = _load_module_dynamically(sector_key)
    if not module:
        return f"No specialist found for sector: {sector}"

    # In a real scenario, we'd instantiate the agent class from the module
    # For now, we return a routing confirmation
    return f"Routed to {module.__name__} for {sector} analysis."

@mcp.tool()
async def generate_code(spec: str) -> str:
    """
    Uses the CoderAgent to generate code based on a specification.
    """
    module = _load_module_dynamically("coder")
    if not module:
        return "CoderAgent not available."

    # Placeholder for agent invocation
    return f"CoderAgent received spec: {spec[:50]}... (Code generation started)"

@mcp.tool()
async def review_code(code: str) -> str:
    """
    Uses the ReviewerAgent to review the provided code.
    """
    module = _load_module_dynamically("reviewer")
    if not module:
        return "ReviewerAgent not available."

    # Placeholder for agent invocation
    return "ReviewerAgent: Code looks solid. No critical vulnerabilities found (Mock)."

@mcp.tool()
async def run_autonomous_improvement() -> str:
    """
    Triggers the MIT SEAL Autonomous Improvement Loop.
    """
    module = _load_module_dynamically("seal_loop")
    if not module or not hasattr(module, "SEALLoop"):
        return "SEAL Loop module not available."

    seal = module.SEALLoop()
    result = await seal.run_autonomous_improvement(iterations=1)
    return json.dumps(result, indent=2)


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
