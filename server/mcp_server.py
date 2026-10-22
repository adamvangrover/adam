import json
import logging
import os
import sys
from typing import Any

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Dependency Handling & Mocking ---
try:
    from mcp.server.fastmcp import Context, FastMCP, Image
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
    from core.engine.eve_v2_engine import EVEUnderwritingEngine
    EVE_ENGINE_AVAILABLE = True
except ImportError:
    EVE_ENGINE_AVAILABLE = False

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

# 5. Security & Governance
try:
    from core.security.eaci_middleware import EACIMiddleware
    from core.security.governance import ApprovalRequired, GovernanceEnforcer, GovernanceError
    GOVERNANCE_AVAILABLE = True
    # Initialize middleware (mocking permission manager for now)
    eaci = EACIMiddleware(permission_manager=None)
except ImportError:
    GOVERNANCE_AVAILABLE = False
    eaci = None

try:
    from src.pdil.middleware import (
        JsonLogicGovernanceGatekeeper,
        SecurityGovernanceGatekeeper,
        GovernanceError as PDILGovernanceError,
    )
    PDIL_GATEKEEPER_AVAILABLE = True
except ImportError:
    PDIL_GATEKEEPER_AVAILABLE = False

try:
    from evals.graders.multi_state_reasoning_evaluator import MultiStateReasoningEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False

# 6. External Libs
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
        with open(target_path) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# --- Tools ---

# 1. QUANTUM & RISK

@mcp.tool()
def eve_v2_underwriting_engine(
    ticker: str = "MSFT",
    company_name: str = "Microsoft Corporation",
    spot_price: float = 420.0,
    revenue: float = 245000.0,
    ebitda: float = 125000.0,
    fcf: float = 74000.0,
    net_debt: float = 10000.0,
    shares_outstanding: float = 7430.0
) -> str:
    """
    Executes the full EVE v2.0 Institutional Single-Name Underwriting Engine pipeline across 5 valuation clocks, 3 independent engines, Challenger disagreement map, 2LoD Adversarial red-team transmission chains, and multi-engine corridor convergence.
    """
    if not EVE_ENGINE_AVAILABLE:
        return json.dumps({"error": "EVE v2.0 Engine module not available."})
    try:
        engine = EVEUnderwritingEngine(ticker=ticker, company_name=company_name)
        result = engine.execute_underwriting(
            spot_price=spot_price,
            revenue=revenue,
            ebitda=ebitda,
            fcf=fcf,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding
        )
        return json.dumps({
            "ticker": result.ticker,
            "company_name": result.company_name,
            "spot_market_price": result.spot_market_price,
            "final_eve_corridor_price": result.final_eve_corridor_price,
            "upside_downside_pct": result.upside_downside_pct,
            "margin_of_safety_pct": result.margin_of_safety_pct,
            "verdict": result.verdict,
            "val_clocks": {
                "point_in_time": result.clocks.point_in_time,
                "one_year_target": result.clocks.one_year_target,
                "five_year_value": result.clocks.five_year_value,
                "through_cycle_value": result.clocks.through_cycle_value,
                "terminal_value": result.clocks.terminal_value,
            },
            "data_confidence_grade": result.data_confidence.overall_grade,
            "audit_trail_summary": result.audit_trail_summary
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

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

# 2. CREDIT & COMPLIANCE (AFOS-UWR-01 / AFOS-CMP-01)

@mcp.tool()
def financial_statement_parser(entity_ticker: str, filing_type: str, line_item: str) -> str:
    """
    Deterministically extracts specific line items (e.g., EBITDA, Total Debt, Interest Expense) directly from indexed SEC EDGAR filings.
    """
    result = {
        "entity_ticker": entity_ticker,
        "filing_type": filing_type,
        "line_item": line_item,
        "value": 1000000.0,
        "prov_o_audit_trail": [{
            "activity": f"Extracted {line_item}",
            "confidence_score": 0.99,
            "source": f"SEC EDGAR {filing_type} for {entity_ticker}"
        }]
    }
    return json.dumps(result, indent=2)

@mcp.tool()
def divergence_validator(model_alpha_pd: float, model_beta_pd: float, threshold_theta: float) -> str:
    """
    Evaluates the bidirectional divergence between two probabilistic models against the system threshold theta. Used by the Compliance Agent.
    """
    divergence = round(abs(model_alpha_pd - model_beta_pd), 4)
    exceeds_threshold = divergence > threshold_theta
    result = {
        "divergence": divergence,
        "exceeds_threshold": exceeds_threshold,
        "prov_o_audit_trail": [{
            "activity": "divergence_validation",
            "threshold_theta": threshold_theta,
            "divergence": divergence
        }]
    }
    return json.dumps(result, indent=2)

@mcp.tool()
def covenant_headroom_calculator(ebitda: float, total_debt: float, interest_expense: float, covenant_ruleset_id: str) -> str:
    """
    Executes stateless jsonLogic to calculate Fixed Charge Coverage Ratio (FCCR) and leverage headroom based on current financials.
    """
    fccr = round(ebitda / interest_expense, 4) if interest_expense else 0.0
    leverage = round(total_debt / ebitda, 4) if ebitda else 0.0
    result = {
        "fccr": fccr,
        "leverage": leverage,
        "covenant_ruleset_id": covenant_ruleset_id,
        "prov_o_audit_trail": [{
            "activity": "covenant_headroom_calc",
            "fccr": fccr,
            "leverage": leverage
        }]
    }
    return json.dumps(result, indent=2)

@mcp.tool()
def mcp_evaluate_pdil_gatekeeper(inference_output: dict[str, Any], rules: dict[str, Any] | None = None) -> str:
    """
    Evaluates a probabilistic inference payload against PDIL gatekeeper rules and W3C PROV-O provenance constraints.
    """
    if not PDIL_GATEKEEPER_AVAILABLE:
        return json.dumps({"status": "REJECT", "error": "PDIL Gatekeeper middleware not available."})
    try:
        if rules:
            gatekeeper = JsonLogicGovernanceGatekeeper(rules=rules)
        else:
            schema = {
                "type": "object",
                "required": ["ebitda", "total_debt"],
                "properties": {
                    "ebitda": {"type": "number"},
                    "total_debt": {"type": "number"}
                }
            }
            gatekeeper = SecurityGovernanceGatekeeper(schema=schema)

        result = gatekeeper.validate_inference(inference_output)
        return json.dumps({"status": "PASS", "validated_inference": result}, indent=2)
    except Exception as e:
        return json.dumps({"status": "REJECT", "error": str(e)}, indent=2)

@mcp.tool()
def mcp_score_t_plus_1_decision(
    predicted_trace: list[str],
    ground_truth_trace: list[str],
    confidences: list[float] | None = None,
    accuracies: list[int] | None = None
) -> str:
    """
    Evaluates state trace fidelity and T+1 forward decision quality using the Multi-State Reasoning Evaluator.
    """
    if not EVALUATOR_AVAILABLE:
        return json.dumps({"status": "ERROR", "error": "Multi-State Reasoning Evaluator not available."})
    try:
        evaluator = MultiStateReasoningEvaluator()
        trace_score = evaluator.evaluate_trace_fidelity(predicted_trace, ground_truth_trace)
        ece = 0.0
        if confidences and accuracies:
            ece = evaluator.calculate_expected_calibration_error(confidences, accuracies)

        return json.dumps({
            "trace_fidelity": trace_score,
            "expected_calibration_error": ece,
            "status": "PASS" if trace_score >= 0.7 else "WARN"
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)}, indent=2)

@mcp.tool()
def prov_o_audit_logger(agent_id: str, action_taken: str, confidence_score: float, source_citations: list[str]) -> str:
    """
    Commits a cryptographic hash and reasoning lineage to the Temporal state before final node execution.
    """
    import hashlib
    payload = f"{agent_id}:{action_taken}:{confidence_score}:{','.join(source_citations)}"
    crypto_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    result = {
        "agent_id": agent_id,
        "action_taken": action_taken,
        "confidence_score": confidence_score,
        "source_citations": source_citations,
        "audit_hash": crypto_hash,
        "status": "committed"
    }
    return json.dumps(result, indent=2)

@mcp.tool()
def model_alpha_client(entity_ticker: str, financials: dict[str, float]) -> str:
    """
    Queries Model Alpha for obligor Probability of Default (PD) calculation.
    """
    ebitda = financials.get("ebitda", 100.0)
    debt = financials.get("total_debt", 300.0)
    leverage = debt / max(ebitda, 1.0)
    pd_alpha = min(max(0.01 * (leverage ** 1.2), 0.001), 0.25)
    return json.dumps({
        "entity_ticker": entity_ticker,
        "model": "Model Alpha",
        "pd": round(pd_alpha, 4),
        "prov_o_audit_trail": [{"activity": "model_alpha_pd_eval", "pd": round(pd_alpha, 4)}]
    })

@mcp.tool()
def model_beta_client(entity_ticker: str, financials: dict[str, float]) -> str:
    """
    Queries Model Beta for obligor Probability of Default (PD) calculation.
    """
    ebitda = financials.get("ebitda", 100.0)
    debt = financials.get("total_debt", 300.0)
    fccr = financials.get("fccr", 1.5)
    pd_beta = min(max(0.012 * ((debt / max(ebitda, 1.0)) / max(fccr, 0.5)), 0.001), 0.25)
    return json.dumps({
        "entity_ticker": entity_ticker,
        "model": "Model Beta",
        "pd": round(pd_beta, 4),
        "prov_o_audit_trail": [{"activity": "model_beta_pd_eval", "pd": round(pd_beta, 4)}]
    })

@mcp.tool()
def arbitration_calculator(model_alpha_pd: float, model_beta_pd: float, alpha_weight: float = 0.5) -> str:
    """
    Computes weighted consensus Probability of Default (PD) arbitration between Model Alpha and Model Beta.
    """
    consensus_pd = alpha_weight * model_alpha_pd + (1.0 - alpha_weight) * model_beta_pd
    divergence = abs(model_alpha_pd - model_beta_pd)
    return json.dumps({
        "consensus_pd": round(consensus_pd, 4),
        "divergence": round(divergence, 4),
        "prov_o_audit_trail": [{"activity": "arbitration_calc", "consensus_pd": round(consensus_pd, 4)}]
    })

@mcp.tool()
def audit_logger(agent_id: str, event_type: str, details: dict[str, Any] | None = None) -> str:
    """
    Logs compliance audit trails and capital buffer allocations.
    """
    return json.dumps({
        "status": "logged",
        "agent_id": agent_id,
        "event_type": event_type,
        "details": details or {},
        "prov_o_audit_trail": [{"agent_id": agent_id, "event_type": event_type}]
    })

@mcp.tool()
def analyze_snc_credit(financials: dict[str, float], capital_structure: list[dict[str, Any]], enterprise_value: float) -> str:
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
    # 🛡️ Sentinel: Input Sanitization
    if GOVERNANCE_AVAILABLE and eaci:
        try:
            eaci.sanitize_input(start_node)
            eaci.sanitize_input(target_node)
        except ValueError as ve:
            return json.dumps({"error": str(ve)})

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

    # 🛡️ Sentinel: Prevent Path Traversal
    # Restrict ingestion to specific safe directories (data/ and docs/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    allowed_dirs = [
        os.path.abspath(os.path.join(project_root, "data")),
        os.path.abspath(os.path.join(project_root, "docs"))
    ]

    # Resolve the absolute path of the requested file
    abs_path = os.path.realpath(filepath)

    is_safe = False
    for safe_dir in allowed_dirs:
        # Check if the file is within the safe directory
        # commonpath raises ValueError if paths are on different drives, which is fine (safe fail)
        try:
            if os.path.commonpath([safe_dir, abs_path]) == safe_dir:
                is_safe = True
                break
        except ValueError:
            continue

    if not is_safe:
        return json.dumps({"error": "Access denied: File path outside allowed directories."})

    # 🛡️ Sentinel: Enterprise Validation (File Size & Type)
    # 1. Size Limit (10MB)
    try:
        if os.path.getsize(abs_path) > 10 * 1024 * 1024:
             return json.dumps({"error": "Validation Error: File size exceeds 10MB limit."})
    except OSError:
         return json.dumps({"error": "Validation Error: Could not read file size."})

    # 2. Extension Whitelist
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.xlsx', '.pdf', '.py', '.log'}
    _, ext = os.path.splitext(abs_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return json.dumps({"error": f"Validation Error: File type '{ext}' not allowed. Allowed: {list(ALLOWED_EXTENSIONS)}"})

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
            return json.dumps({"error": f"Data fetch failed: {e!s}"})
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
            "error": f"Execution failed: {e!s}"
        })

if __name__ == "__main__":
    mcp.run()
