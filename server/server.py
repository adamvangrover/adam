try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback for environments without MCP installed
    class FastMCP:
        def __init__(self, name): self.name = name
        def resource(self, path): return lambda f: f
        def tool(self): return lambda f: f
        def prompt(self, name): return lambda f: f
        def run(self): print("MCP Server Mock Run (Dependencies missing)")

import sqlite3
import pandas as pd
import sys
import os
import asyncio
import json
import math
import io
from contextlib import redirect_stdout
from typing import List, Dict, Any, Optional

# Ensure core is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try importing yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None

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

# Import Credit Sentinel
try:
    from core.credit_sentinel.agents.ratio_calculator import RatioCalculator
    from core.credit_sentinel.models.distress_classifier import DistressClassifier
    from core.credit_sentinel.agents.risk_analyst import RiskAnalyst, AgentInput
    CREDIT_SENTINEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Credit Sentinel import failed: {e}")
    CREDIT_SENTINEL_AVAILABLE = False

# NEW IMPORT
try:
    from core.agents.risk_assessment_agent import RiskAssessmentAgent as System2RiskAgent
except ImportError:
    System2RiskAgent = None

try:
    from core.governance.constitution import Constitution
    CONSTITUTION_AVAILABLE = True
except ImportError:
    CONSTITUTION_AVAILABLE = False

try:
    from core.security.governance import GovernanceEnforcer, GovernanceError, ApprovalRequired
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False


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


@mcp.resource("adam://profile/bio")
def get_profile_bio() -> str:
    """
    Returns the professional biography of Adam.
    Resource URI: adam://profile/bio
    """
    return """
    Adam is a financial services professional with a background in credit risk management,
    investment banking, and corporate ratings advisory. He has experience at institutions such as
    Credit Suisse and S&P. He specializes in quantitative finance, agentic AI systems, and
    risk modeling.
    """


@mcp.resource("adam://portfolio/case-studies")
def get_case_studies() -> str:
    """
    Returns selected MOCK financial case studies.
    Resource URI: adam://portfolio/case-studies
    """
    return json.dumps([
        {"title": "MOCK LBO of Tech Conglomerate", "roi": "22%", "sector": "Technology"},
        {"title": "MOCK Distressed Debt Restructuring", "recovery": "85 cents on dollar", "sector": "Energy"},
        {"title": "MOCK IPO Advisory for FinTech", "valuation": "$2.5B", "sector": "Finance"}
    ], indent=2)


@mcp.resource("adam://docs/architecture")
def get_architecture_docs() -> str:
    """
    Returns the high-level system architecture documentation.
    Resource URI: adam://docs/architecture
    """
    try:
        # Try to read the actual architecture doc if it exists, else return summary
        with open("docs/v23_architecture_vision.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        Adam v23.5 Adaptive System Architecture:
        1. MetaOrchestrator: Central routing brain.
        2. Cyclical Reasoning Graphs: LangGraph-based loops (Red Team, Crisis Sim).
        3. MCP Server: Standardized tool exposure.
        4. Hyper-Dimensional Knowledge Graph (HDKG): Unified data structure.
        """

# --- Tools ---

@mcp.tool()
async def analyze_distressed_debt(ticker: str) -> str:
    """
    Analyzes a company for signs of financial distress using the Credit Sentinel module.
    Returns a comprehensive JSON report including ratios, distress probability, and qualitative analysis.
    """
    try:
        # 1. Mock Data Fetch (Replace with UniversalIngestor in prod)
        financials = {
            'ebitda': 500,
            'total_debt': 2500,
            'interest_expense': 300,
            'total_assets': 5000,
            'total_liabilities': 3000,
            'total_equity': 2000,
            'current_assets': 1000,
            'current_liabilities': 800,
            'net_income': 100,
            'cash': 200,
            'monthly_burn_rate': 20,
            'industry': 'Technology',
            'credit_rating': 'B-'
        }

        # Upgrade: Use System 2 RiskAssessmentAgent if available
        if System2RiskAgent:
            agent = System2RiskAgent(config={"persona": "Credit Underwriter"})
            target_data = {
                "company_name": ticker,
                "financial_data": financials,
                "market_data": {"price_data": [100, 90, 80], "trading_volume": 500000}
            }
            # Execute Async
            result = await agent.execute(target_data, risk_type="investment")
            return json.dumps(result, indent=2, default=str)

        # Fallback to Legacy CreditSentinel
        if not CREDIT_SENTINEL_AVAILABLE:
            return json.dumps({"error": "Credit Sentinel module not available."})

        # 2. Calculate Ratios
        calc = RatioCalculator()
        ratios = calc.calculate_all(financials)

        # 3. Predict Distress
        classifier = DistressClassifier()
        prediction = classifier.predict_distress(ratios)

        # 4. Qualitative Analysis (Agent)
        analyst = RiskAnalyst()
        context = {
            "financials": financials,
            "ratios": ratios,
            "distress_prediction": prediction
        }
        agent_input = AgentInput(query=f"Analyze distress risk for {ticker}", context=context)
        report = analyst.execute(agent_input)

        return report.model_dump_json(indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def run_stress_test(portfolio_id: str, scenario: str) -> str:
    """
    Triggers a System 2 Stress Test via MetaOrchestrator.
    """
    orchestrator = get_orchestrator_instance()
    if not orchestrator:
        return "Error: MetaOrchestrator not available."

    query = f"Run stress test for {portfolio_id} under {scenario} scenario."
    # We force the route to CRISIS or HIGH
    context = {"force_route": "CRISIS", "portfolio_id": portfolio_id}

    try:
        result = await orchestrator.route_request(query, context)
        return str(result)
    except Exception as e:
        return f"Error running stress test: {str(e)}"

@mcp.tool()
def query_historical_defaults(industry: str, rating: str) -> str:
    """
    Queries historical default rates from the 'Logic as Data' repository.
    """
    # Mock Database
    defaults = {
        "Technology": {"AAA": 0.0, "BBB": 0.05, "CCC": 0.25, "B-": 0.15},
        "Energy": {"AAA": 0.01, "BBB": 0.08, "CCC": 0.35, "B-": 0.20}
    }
    rate = defaults.get(industry, {}).get(rating, "N/A")
    return json.dumps({"industry": industry, "rating": rating, "historical_default_rate": rate})


@mcp.tool()
def calculate_credit_exposure(entity_id: str, amount: float) -> str:
    """
    Calculates credit exposure for a given entity and amount.
    Wraps internal risk logic.
    """
    # Logic placeholder - in a real scenario this would call a risk engine
    # For now, we simulate a risk factor calculation
    risk_factor = 0.15  # Mock risk factor (e.g. 15%)

    # Adjust risk factor based on entity for demo purposes
    if entity_id.upper() in ["AMC", "GME"]:
        risk_factor = 0.45
    elif entity_id.upper() in ["AAPL", "MSFT"]:
        risk_factor = 0.05

    exposure = amount * risk_factor
    return json.dumps({
        "entity_id": entity_id,
        "amount": amount,
        "risk_factor": risk_factor,
        "credit_exposure": exposure,
        "currency": "USD",
        "methodology": "Standardized Approach"
    })

@mcp.tool()
def retrieve_market_data(ticker: str) -> str:
    """
    Retrieves real-time market data for a ticker using yfinance.
    """
    if yf is None:
        return json.dumps({"error": "yfinance module not available"})

    try:
        ticker_obj = yf.Ticker(ticker)
        # We fetch info, but handle potential connection errors gracefully
        info = ticker_obj.info

        # Extract key fields to avoid overwhelming context
        data = {
            "symbol": info.get("symbol"),
            "currentPrice": info.get("currentPrice"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow")
        }
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving data for {ticker}: {str(e)}"})

@mcp.tool()
def execute_python_sandbox(code: str) -> str:
    """
    Executes Python code in a secure, isolated sandbox.

    Security Features:
    - Governance Enforcer: Checks for risk patterns (recursion, loops, imports).
    - Static Analysis (AST) prevents dangerous imports and access to private attributes.
    - Process Isolation contains memory and crashes.
    - Restricted Globals limits accessible functions to safe subsets (math, json, pandas).
    - Timeouts prevent infinite loops.
    """
    try:
        # 🛡️ Sentinel: Governance Enforcement
        if GOVERNANCE_AVAILABLE:
            try:
                GovernanceEnforcer.validate(code, context="mcp_tool")
            except ApprovalRequired as ae:
                return json.dumps({
                    "status": "approval_required",
                    "error": str(ae),
                    "score": GovernanceEnforcer.analyze_risk(code).get("score")
                })
            except GovernanceError as ge:
                return json.dumps({
                    "status": "blocked",
                    "error": str(ge)
                })

        from core.security.sandbox import SecureSandbox
        result = SecureSandbox.execute(code)
        return json.dumps(result)
    except ImportError:
        return json.dumps({
            "status": "error",
            "error": "SecureSandbox module not found. Please contact administrator."
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"Execution failed: {str(e)}"
        })

@mcp.tool()
def execute_market_order(symbol: str, quantity: int, side: str) -> str:
    """
    Executes a market order. Requires Human-in-the-Loop confirmation.
    """
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
    return f"Based on your personal memory: You have a preference for 'Low Volatility' stocks. Regarding '{query}': We found relevant notes on this topic."


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

    ALLOWED_TABLES = {'financials', 'sqlite_sequence'}

    def authorizer(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    try:
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
async def run_red_team_attack(scenario: str) -> str:
    """
    Executes a Red Team adversarial attack on a financial model or portfolio.
    Uses the v23 RedTeamGraph.
    """
    # Placeholder for actual graph execution
    return json.dumps({
        "status": "ATTACK_COMPLETE",
        "scenario": scenario,
        "vulnerabilities_found": [
            "Liquidity crunch in low-volatility regime",
            "Correlation breakdown in Tech sector"
        ],
        "severity": "HIGH"
    }, indent=2)


@mcp.tool()
async def run_crisis_simulation(macro_shock: str) -> str:
    """
    Runs a macro-economic crisis simulation.
    Uses the v23 CrisisSimulationGraph.
    """
    # Placeholder for actual graph execution
    return json.dumps({
        "status": "SIMULATION_COMPLETE",
        "shock": macro_shock,
        "impact": {
            "sp500": "-15%",
            "high_yield_spreads": "+400bps",
            "gdp_growth": "-1.2%"
        }
    }, indent=2)

if __name__ == "__main__":
    # Ensure DB exists
    if not os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS financials
                         (ticker text, year int, revenue real, ebitda real)''')
            c.execute("INSERT INTO financials VALUES ('AAPL', 2023, 383285, 114301)")
            conn.commit()
            conn.close()
        except Exception:
            pass # Ignore DB creation errors if we don't have write permissions or other issues

    mcp.run()
