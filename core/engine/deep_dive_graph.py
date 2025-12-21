import logging
from typing import Any, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.engine.entity_utils import assess_competitive_position, assess_management
from core.engine.snc_utils import calculate_leverage, map_financials_to_rating
from core.engine.states import OmniscientState
from core.engine.strategy_utils import determine_ma_posture, synthesize_verdict
from core.engine.valuation_utils import calculate_dcf, calculate_multiples, get_price_targets
from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine

logger = logging.getLogger(__name__)

# --- Mock Data Fetcher ---

def fetch_financial_context(ticker: str) -> Dict[str, Any]:
    """
    Simulates fetching diverse datasets from a data lake (SNC, Market, Fundamentals).
    """
    # Base numbers for simulation
    return {
        "fundamentals": {
            "revenue": 10000,
            "ebitda": 2500,
            "fcf": 1500,
            "total_debt": 5000,
            "cash_equivalents": 800,
            "shares_outstanding": 500,
            "beta": 1.1,
            "growth_rate": 0.04,
            "tax_rate": 0.21,
            "net_debt": 4200,
            "enterprise_value": 30000
        },
        "market_data": {
            "market_cap": 25000,
            "current_price": 50.0,
            "volatility": 0.25,
            "pe_ratio": 15.0
        },
        "syndicate_data": {
            "facilities": [
                {"id": "TLB", "amount": 2000, "rate": "S+350"},
                {"id": "RCF", "amount": 500, "rate": "S+250"}
            ],
            "banks": [{"role": "Lead", "share": 0.15}]
        },
        "peers": [
            {"ticker": "COMP1", "ev_ebitda": 11.5},
            {"ticker": "COMP2", "ev_ebitda": 12.8}
        ]
    }

# --- Nodes ---

def entity_resolution_node(state: OmniscientState) -> Dict[str, Any]:
    """
    Phase 1: Entity, Ecosystem & Management.
    """
    logger.info(f"Phase 1: Analyzing Entity {state['ticker']}")
    ticker = state["ticker"]

    # In reality, verify legal entity hierarchy here.
    legal_entity = {
        "name": f"{ticker} Global Holdings Inc.",
        "lei": "5493006MHB84DD0ZWV18", # Mock LEI
        "jurisdiction": "Delaware, USA"
    }

    mgmt = assess_management(ticker)
    comp = assess_competitive_position(ticker, "Technology") # Assume tech for now

    # Update State
    legal_entity["sector"] = "Technology" # Default or extracted

    state["v23_knowledge_graph"]["nodes"]["entity_ecosystem"] = {
        "legal_entity": legal_entity,
        "management_assessment": mgmt,
        "competitive_positioning": comp
    }

    return {
        "v23_knowledge_graph": state["v23_knowledge_graph"],
        "human_readable_status": "Phase 1 Complete: Entity & Management Assessed."
    }

def deep_fundamental_node(state: OmniscientState) -> Dict[str, Any]:
    """
    Phase 2: Deep Fundamental & Valuation.
    """
    logger.info("Phase 2: Valuation Models")
    data = fetch_financial_context(state["ticker"])
    fin = data["fundamentals"]
    peers = data["peers"]

    # 1. DCF
    dcf_res = calculate_dcf(fin, risk_free_rate=0.042)

    # 2. Multiples
    mult_res = calculate_multiples(fin, peers)

    # 3. Price Targets
    targets = get_price_targets(dcf_res["intrinsic_share_price"], data["market_data"]["volatility"])

    # Fundamentals Summary
    fund_summary = {
        "revenue_cagr_3yr": "5.2%",
        "ebitda_margin_trend": "Expanding"
    }

    # Adapter for DCF
    dcf_res_adapter = {
        "wacc_assumption": f"{dcf_res.get('wacc', 0.08):.1%}",
        "terminal_growth": f"{dcf_res.get('terminal_growth', 0.02):.1%}",
        "intrinsic_value_estimate": dcf_res.get('intrinsic_share_price', 0.0) if 'intrinsic_share_price' in dcf_res else dcf_res.get('intrinsic_value', 0.0)
    }

    # Adapter for Multiples
    mult_res_adapter = {
        "current_ev_ebitda": mult_res.get("current_ev_ebitda", 0.0),
        "peer_median_ev_ebitda": mult_res.get("peer_median_ev_ebitda", 0.0),
        "verdict": "Undervalued" # Mock logic or derived
    }

    state["v23_knowledge_graph"]["nodes"]["equity_analysis"] = {
        "fundamentals": fund_summary,
        "valuation_engine": {
            "dcf_model": dcf_res_adapter,
            "multiples_analysis": mult_res_adapter,
            "price_targets": targets
        }
    }

    return {
        "v23_knowledge_graph": state["v23_knowledge_graph"],
        "human_readable_status": "Phase 2 Complete: Valuation Models Run."
    }

def credit_snc_node(state: OmniscientState) -> Dict[str, Any]:
    """
    Phase 3: Credit, Covenants & SNC Ratings.
    """
    logger.info("Phase 3: Credit & SNC")
    data = fetch_financial_context(state["ticker"])
    fin = data["fundamentals"]

    leverage = calculate_leverage(fin["total_debt"], fin["ebitda"])
    liquidity = fin["cash_equivalents"]

    rating = map_financials_to_rating(leverage, liquidity, fin["total_debt"])

    # Primary Facility Logic
    primary_fac = data["syndicate_data"]["facilities"][0] if data["syndicate_data"]["facilities"] else {"id": "N/A", "amount": 0}

    primary_facility_assessment = {
        "facility_type": primary_fac.get("id", "Term Loan"),
        "collateral_coverage": "Strong", # Mock
        "repayment_capacity": "High" if rating == "Pass" else "Low"
    }

    snc_model = {
        "overall_borrower_rating": rating,
        "rationale": "Strong coverage ratios support rating.",
        "primary_facility_assessment": primary_facility_assessment
    }

    covenant_risk = {
        "primary_constraint": "Max Net Leverage 4.5x",
        "current_level": leverage,
        "breach_threshold": 4.5,
        "headroom_assessment": "Low" if leverage > 4.0 else "High"
    }

    state["v23_knowledge_graph"]["nodes"]["credit_analysis"] = {
        "snc_rating_model": snc_model,
        "cds_market_implied_rating": "BB-", # Mock
        "covenant_risk_analysis": covenant_risk
    }

    return {
        "v23_knowledge_graph": state["v23_knowledge_graph"],
        "human_readable_status": "Phase 3 Complete: Credit Rating Assigned."
    }

def risk_simulation_node(state: OmniscientState) -> Dict[str, Any]:
    """
    Phase 4: Risk, Simulation & Quantum Modeling.
    """
    logger.info("Phase 4: Simulation Engine")

    # Initialize Generative Risk Engine
    # In a real run, we might pass a path, here we use default mock
    engine = GenerativeRiskEngine()

    # Generate Tail Scenarios
    scenarios = engine.generate_scenarios(n_samples=5, regime="stress")

    quantum_scenarios = []
    for s in scenarios:
        prob = "Low"
        if s.probability_weight > 0.4: prob = "High"
        elif s.probability_weight > 0.1: prob = "Med"

        quantum_scenarios.append({
            "scenario_name": s.description,
            "probability": prob,
            "impact_severity": "High",
            "estimated_impact_ev": "-15%" # Simplified
        })

    simulation_engine = {
        "monte_carlo_default_prob": "2.5%",
        "quantum_scenarios": quantum_scenarios,
        "trading_dynamics": {
            "short_interest": "3.2%",
            "liquidity_risk": "Low"
        }
    }

    state["v23_knowledge_graph"]["nodes"]["simulation_engine"] = simulation_engine

    return {
        "v23_knowledge_graph": state["v23_knowledge_graph"],
        "human_readable_status": "Phase 4 Complete: Stress Tests Run."
    }

def strategic_synthesis_node(state: OmniscientState) -> Dict[str, Any]:
    """
    Phase 5: Synthesis, Conviction & Strategy.
    """
    logger.info("Phase 5: Final Synthesis")
    data = fetch_financial_context(state["ticker"])
    kg = state["v23_knowledge_graph"]["nodes"]

    # Extract inputs for synthesis
    val_verdict = kg["equity_analysis"]["valuation_engine"]["multiples_analysis"]
    credit_rating = kg["credit_analysis"]["snc_rating_model"]["overall_borrower_rating"]
    pd_str = kg["simulation_engine"]["monte_carlo_default_prob"]

    # Parse PD string "2.5%" -> 0.025
    try:
        if isinstance(pd_str, str):
            pd = float(pd_str.replace("%", "")) / 100.0
        else:
            pd = float(pd_str)
    except:
        pd = 0.05 # Default if parsing fails

    # 1. M&A Posture
    ma_posture = determine_ma_posture(data["fundamentals"], data["market_data"])

    # 2. Final Verdict
    # Normalize PD to risk score 0-10 (0.05 -> 10)
    risk_score = pd * 200

    verdict = synthesize_verdict(
        valuation=val_verdict,
        credit_rating=credit_rating,
        risk_score=risk_score,
        ma_posture=ma_posture
    )

    state["v23_knowledge_graph"]["nodes"]["strategic_synthesis"] = {
        "m_and_a_posture": ma_posture,
        "final_verdict": verdict
    }

    human_msg = f"Analysis Complete. Recommendation: {verdict['recommendation']} (Conviction {verdict['conviction_level']}/10)."

    return {
        "v23_knowledge_graph": state["v23_knowledge_graph"],
        "human_readable_status": human_msg
    }

# --- Graph Construction ---

def build_deep_dive_graph():
    workflow = StateGraph(OmniscientState)

    # Add Nodes
    workflow.add_node("entity_resolution", entity_resolution_node)
    workflow.add_node("deep_fundamental", deep_fundamental_node)
    workflow.add_node("credit_snc", credit_snc_node)
    workflow.add_node("risk_simulation", risk_simulation_node)
    workflow.add_node("strategic_synthesis", strategic_synthesis_node)

    # Add Edges (Linear Flow)
    workflow.add_edge(START, "entity_resolution")
    workflow.add_edge("entity_resolution", "deep_fundamental")
    workflow.add_edge("deep_fundamental", "credit_snc")
    workflow.add_edge("credit_snc", "risk_simulation")
    workflow.add_edge("risk_simulation", "strategic_synthesis")
    workflow.add_edge("strategic_synthesis", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

deep_dive_app = build_deep_dive_graph()
