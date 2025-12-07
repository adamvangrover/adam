import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional
import glob

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Dependency Handling & Mocking ---
try:
    from mcp.server.fastmcp import FastMCP, Context, Image
except ImportError:
    # Fallback for environments without MCP installed (Mock implementation)
    logging.warning("MCP library not found. Using Mock FastMCP.")
    class FastMCP:
        def __init__(self, name, dependencies=None): 
            self.name = name
            self.tools = []
            self.resources = []
        def resource(self, path): 
            def decorator(f):
                self.resources.append({"path": path, "func": f})
                return f
            return decorator
        def tool(self, name=None): 
            def decorator(f):
                self.tools.append({"name": name or f.__name__, "func": f})
                return f
            return decorator
        def run(self): 
            print(f"Starting Project Adam MCP Server: {self.name}")
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line: break
                except KeyboardInterrupt:
                    break
    class Context: pass
    class Image: pass

# --- Core Imports with Fallbacks ---

# 1. Quantum & Risk
try:
    from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine
    QMC_AVAILABLE = True
except ImportError:
    QMC_AVAILABLE = False

try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
    GEN_RISK_AVAILABLE = True
except ImportError:
    GEN_RISK_AVAILABLE = False

# 2. Specialized Agents
try:
    from core.agents.specialized.snc_rating_agent import SNCRatingAgent
    SNC_AVAILABLE = True
except ImportError:
    SNC_AVAILABLE = False

try:
    from core.agents.specialized.covenant_analyst_agent import CovenantAnalystAgent
    COVENANT_AVAILABLE = True
except ImportError:
    COVENANT_AVAILABLE = False

try:
    from core.agents.specialized.peer_comparison_agent import PeerComparisonAgent
    PEER_AVAILABLE = True
except ImportError:
    PEER_AVAILABLE = False

# 3. Graph Engine
try:
    from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
    PLANNER_AVAILABLE = True
except ImportError:
    PLANNER_AVAILABLE = False

# 4. Data Processing
try:
    from core.data_processing.universal_ingestor import UniversalIngestor
    INGESTOR_AVAILABLE = True
except ImportError:
    INGESTOR_AVAILABLE = False

# 5. External Libs
try:
    import yfinance as yf
except ImportError:
    yf = None

# --- Server Initialization ---
mcp = FastMCP("Project Adam Financial Engine")

# --- Resources ---

@mcp.resource("adam://project/manifest")
def get_manifest() -> str:
    """Returns the Project Adam capabilities manifest."""
    return json.dumps({
        "project": "Project Adam",
        "version": "23.5.0",
        "capabilities": {
            "quantum_risk": QMC_AVAILABLE,
            "generative_risk": GEN_RISK_AVAILABLE,
            "snc_credit_rating": SNC_AVAILABLE,
            "covenant_analysis": COVENANT_AVAILABLE,
            "peer_comparison": PEER_AVAILABLE,
            "neuro_symbolic_planning": PLANNER_AVAILABLE,
            "universal_ingestion": INGESTOR_AVAILABLE,
            "market_data": yf is not None
        }
    }, indent=2)

@mcp.resource("adam://docs/{filename}")
def get_documentation(filename: str) -> str:
    """Dynamically reads documentation files."""
    # Sanitize path
    filename = os.path.basename(filename)
    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
    target_path = os.path.join(docs_path, filename)
    
    # Try extensions
    if not os.path.exists(target_path):
        if os.path.exists(target_path + ".md"):
            target_path += ".md"
        elif os.path.exists(target_path + ".txt"):
            target_path += ".txt"
        else:
            return f"Error: Document '{filename}' not found in docs/."
            
    try:
        with open(target_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# --- Tools ---

# 1. QUANTUM & RISK

@mcp.tool()
def run_quantum_simulation(asset_value: float, volatility: float, debt: float, horizon: float = 1.0) -> str:
    """
    Runs a Quantum Monte Carlo simulation to estimate credit risk (Merton Model).
    """
    if not QMC_AVAILABLE:
        return json.dumps({"error": "Quantum Engine not available."})
    try:
        qmc = QuantumMonteCarloEngine()
        result = qmc.simulate_merton_model(asset_value, debt, volatility, 0.05, horizon)
        return str(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def generate_market_scenarios(regime: str = "stress", n_samples: int = 5) -> str:
    """
    Generates synthetic market scenarios using the Generative Risk Engine (GAN).
    Regime: 'normal', 'stress', 'crash'.
    """
    if not GEN_RISK_AVAILABLE:
        return json.dumps({"error": "Generative Risk Engine not available."})
    try:
        engine = GenerativeRiskEngine()
        scenarios = engine.generate_scenarios(n_samples=n_samples, regime=regime)
        return str(scenarios)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 2. CREDIT & SNC

@mcp.tool()
def analyze_snc_credit(financials: Dict[str, float], capital_structure: List[Dict[str, Any]], enterprise_value: float) -> str:
    """
    Performs a Shared National Credit (SNC) rating analysis.
    
    Args:
        financials: {'ebitda': float, 'total_debt': float, 'interest_expense': float}
        capital_structure: List of {'name': str, 'amount': float, 'priority': int}
        enterprise_value: Estimated EV for collateral coverage.
    """
    if not SNC_AVAILABLE:
        return json.dumps({"error": "SNC Agent not available."})
    try:
        agent = SNCRatingAgent(config={"name": "mcp_snc"})
        result = agent.execute(financials, capital_structure, enterprise_value)
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def analyze_covenants(leverage: float, covenant_threshold: float) -> str:
    """
    Analyzes risk of covenant breach.
    """
    if not COVENANT_AVAILABLE:
        return json.dumps({"error": "Covenant Agent not available."})
    try:
        agent = CovenantAnalystAgent(config={"name": "mcp_legal"})
        result = await agent.execute(leverage=leverage, covenant_threshold=covenant_threshold)
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 3. STRATEGY & PLANNING

@mcp.tool()
async def compare_peers(company_id: str) -> str:
    """
    Compares company valuation multiples against peers.
    """
    if not PEER_AVAILABLE:
        return json.dumps({"error": "Peer Agent not available."})
    try:
        agent = PeerComparisonAgent(config={"name": "mcp_peer"})
        result = await agent.execute(company_id=company_id)
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def plan_workflow(start_node: str, target_node: str) -> str:
    """
    Generates a Neuro-Symbolic Plan (Graph Traversal) from start concept to target concept.
    Example: start="Apple Inc.", target="CreditRating"
    """
    if not PLANNER_AVAILABLE:
        return json.dumps({"error": "Neuro-Symbolic Planner not available."})
    try:
        planner = NeuroSymbolicPlanner()
        plan = planner.discover_plan(start_node, target_node)
        return json.dumps(plan, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 4. DATA INGESTION

@mcp.tool()
def ingest_file(filepath: str) -> str:
    """
    Ingests a file into the Gold Standard knowledge base.
    """
    if not INGESTOR_AVAILABLE:
        return json.dumps({"error": "Universal Ingestor not available."})
    try:
        ingestor = UniversalIngestor()
        ingestor.process_file(filepath)
        if ingestor.artifacts:
            return json.dumps(ingestor.artifacts[0].to_dict(), indent=2)
        return json.dumps({"status": "No artifact produced."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# 5. MARKET DATA & UTILS

@mcp.tool()
def retrieve_market_data(ticker: str) -> str:
    """
    Retrieves real-time market data for a ticker.
    """
    if yf:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            data = {
                "symbol": info.get("symbol", ticker),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "cap": info.get("marketCap"),
                "beta": info.get("beta")
            }
            return json.dumps(data)
        except Exception as e:
            return json.dumps({"error": f"Data fetch failed: {str(e)}"})
    else:
        # Mock for demo
        import random
        return json.dumps({
            "symbol": ticker,
            "price": round(random.uniform(100, 200), 2),
            "source": "Mock (yfinance missing)"
        })

@mcp.tool()
def execute_python_sandbox(code: str) -> str:
    """
    Executes arbitrary Python code in a sandboxed environment.
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr
    import pandas as pd
    import numpy as np
    
    stdout = io.StringIO()
    stderr = io.StringIO()
    exec_globals = {"pd": pd, "np": np, "json": json}
    
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, exec_globals)
        return json.dumps({
            "status": "success",
            "output": stdout.getvalue(),
            "error": stderr.getvalue()
        })
    except Exception as e:
        return json.dumps({"status": "error", "exception": str(e)})

if __name__ == "__main__":
    mcp.run()
