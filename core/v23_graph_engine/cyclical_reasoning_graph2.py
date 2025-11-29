# core/v23_graph_engine/cyclical_reasoning_graph2.py

"""
Agent Notes (Meta-Commentary):
This module implements the core cyclical reasoning engine for Adam v23.0.
It replaces the linear v22 simulation with a stateful LangGraph workflow.
It refactors critical logic from v21 agents into a dependency-free structure
suitable for the v23 graph architecture.
"""

import json
import logging
import numpy as np
from typing import Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from core.v23_graph_engine.states import RiskAssessmentState, ResearchArtifact

# Import RiskAssessmentAgent (Math-heavy, no complex dependencies)
try:
    from core.agents.risk_assessment_agent import RiskAssessmentAgent
except ImportError as e:
    logging.error(f"Failed to import RiskAssessmentAgent: {e}")
    RiskAssessmentAgent = None

# Refactored Data Retrieval Logic (Decoupled from Semantic Kernel)
class V23DataRetriever:
    """
    A lightweight, v23-compliant data retriever that mirrors the logic
    of the legacy DataRetrievalAgent but without heavy dependencies.
    """
    def get_financials(self, company_id: str) -> Optional[Dict[str, Any]]:
        # Logic ported from core/agents/data_retrieval_agent.py
        if company_id == "ABC_TEST":
            return {
                "company_info": {
                    "name": f"{company_id} Corp",
                    "industry_sector": "Technology", 
                    "country": "USA"
                },
                "financial_data_detailed": { 
                    "income_statement": { 
                        "revenue": [1000, 1100, 1250], "cogs": [400, 440, 500], "gross_profit": [600, 660, 750],
                        "operating_expenses": [300, 320, 350], "ebitda": [300, 340, 400], "depreciation_amortization": [50, 55, 60],
                        "ebit": [250, 285, 340], "interest_expense": [30, 28, 25], "income_before_tax": [220, 257, 315],
                        "taxes": [44, 51, 63], "net_income": [176, 206, 252]
                    },
                    "balance_sheet": { 
                        "cash_and_equivalents": [200, 250, 300], "accounts_receivable": [150, 160, 170], "inventory": [100, 110, 120],
                        "total_current_assets": [450, 520, 590], "property_plant_equipment_net": [1500, 1550, 1600],
                        "total_assets": [1950, 2070, 2190],
                        "accounts_payable": [120, 130, 140], "short_term_debt": [100, 80, 60], "total_current_liabilities": [220, 210, 200],
                        "long_term_debt": [500, 450, 400], "total_liabilities": [720, 660, 600],
                        "shareholders_equity": [1230, 1410, 1590]
                    },
                    "key_ratios": { 
                        "debt_to_equity_ratio": 0.58, "net_profit_margin": 0.20,
                        "current_ratio": 2.95, "interest_coverage_ratio": 13.6 
                    },
                    "market_data": { 
                        "share_price": 65.00, "shares_outstanding": 10000000 
                    }
                }
            }
        return None

# --- Adapters & Helpers ---

def map_dra_to_raa(financials: Dict[str, Any]) -> tuple[Dict, Dict]:
    """
    Maps DataRetrievalAgent output to RiskAssessmentAgent input.
    """
    fin_details = financials.get("financial_data_detailed", {})
    
    mapped_fin = {
        "industry": financials.get("company_info", {}).get("industry_sector", "Unknown"),
        "credit_rating": "BBB", 
    }
    
    current_price = fin_details.get("market_data", {}).get("share_price", 100)
    mapped_market = {
        "price_data": np.array([current_price * (1 + i*0.01) for i in range(10)]) 
    }
    
    return mapped_fin, mapped_market

# --- Initialization ---

data_retriever = V23DataRetriever()
try:
    # RiskAssessmentAgent might need a valid path for its knowledge base
    # We assume default or it handles missing file gracefully (it does print error but returns dict)
    risk_assessor = RiskAssessmentAgent() if RiskAssessmentAgent else None
except Exception as e:
    logging.error(f"Error init RAA: {e}")
    risk_assessor = None

# --- Nodes ---

def retrieve_data_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Retrieve Data
    Calls V23DataRetriever to fetch financials.
    """
    print("--- Node: Retrieve Data ---")
    ticker = state["ticker"]
    
    artifacts = []
    financials = data_retriever.get_financials(ticker)

    if financials:
        content = json.dumps(financials, indent=2)
        artifacts.append(ResearchArtifact(
            title=f"{ticker} Financial Data",
            content=content,
            source="V23DataRetriever",
            credibility_score=1.0
        ))
        status = f"Retrieved financials for {ticker}."
    else:
        status = f"No data found for {ticker}."

    return {
        "research_data": artifacts,
        "human_readable_status": status
    }

def generate_draft_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Generate Draft
    Calls RiskAssessmentAgent to calculate risk.
    """
    print("--- Node: Generate Draft ---")
    
    if not risk_assessor:
        return {"draft_analysis": "RiskAssessmentAgent unavailable.", "iteration_count": state["iteration_count"] + 1}
    
    financial_data_raw = {}
    for art in state["research_data"]:
        if "Financial Data" in art["title"]:
            try:
                financial_data_raw = json.loads(art["content"])
            except:
                pass
    
    financials, market = map_dra_to_raa(financial_data_raw)
    
    try:
        result = risk_assessor.assess_investment_risk(
            state["ticker"],
            financials,
            market
        )
        
        score = result.get("overall_risk_score", 0)
        factors = result.get("risk_factors", {})
        
        draft = f"RISK ASSESSMENT REPORT FOR {state['ticker']}\n"
        draft += f"Overall Risk Score: {score:.2f}\n\n"
        draft += "Risk Factors:\n"
        for k, v in factors.items():
            draft += f"- {k}: {v}\n"
            
    except Exception as e:
        draft = f"Error during assessment: {e}"

    return {
        "draft_analysis": draft,
        "human_readable_status": "Drafted initial assessment.",
        "iteration_count": state["iteration_count"] + 1
    }

def critique_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Critique
    """
    print("--- Node: Critique ---")
    draft = state.get("draft_analysis", "")
    iteration = state["iteration_count"]
    
    critique_notes = []
    quality_score = 0.0
    needs_correction = False
    
    if "Error" in draft or "unavailable" in draft:
        critique_notes.append("Analysis failed due to system error.")
        quality_score = 0.1
        needs_correction = True
    elif "market_risk" not in draft.lower() and iteration < 3:
        # Note: 'overall_risk_score' is in draft, factors are listed.
        # If RAA works, 'market_risk' should be in the factors.
        critique_notes.append("Missing Market Risk analysis.")
        quality_score = 0.6
        needs_correction = True
    elif iteration < 2:
        critique_notes.append("Refine volatility assessment.")
        quality_score = 0.7
        needs_correction = True
    else:
        critique_notes.append("Assessment looks complete.")
        quality_score = 0.9
        needs_correction = False
        
    return {
        "critique_notes": critique_notes,
        "quality_score": quality_score,
        "needs_correction": needs_correction,
        "human_readable_status": "Critiqued draft."
    }

def correction_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Correction
    """
    print("--- Node: Correction ---")
    draft = state["draft_analysis"]
    notes = state["critique_notes"]
    
    correction_text = f"\n\n[Correction v{state['iteration_count']}]\n" + "\n".join(notes)
    new_draft = draft + correction_text
    
    return {
        "draft_analysis": new_draft,
        "human_readable_status": "Applied corrections.",
        "iteration_count": state["iteration_count"] + 1
    }

def human_review_node(state: RiskAssessmentState) -> Dict[str, Any]:
    print("--- Node: Human Review ---")
    return {
        "human_readable_status": "Awaiting Human Review."
    }

# --- Conditional Logic ---

def should_continue(state: RiskAssessmentState) -> Literal["correct_analysis", "human_review", "END"]:
    quality = state["quality_score"]
    iteration = state["iteration_count"]
    needs_correction = state["needs_correction"]
    
    if not needs_correction or quality >= 0.85:
        return "END"
    
    if iteration >= 4: 
        return "human_review"
        
    return "correct_analysis"

# --- Graph Construction ---

def build_cyclical_reasoning_graph():
    workflow = StateGraph(RiskAssessmentState)
    
    workflow.add_node("retrieve_data", retrieve_data_node)
    workflow.add_node("generate_draft", generate_draft_node)
    workflow.add_node("critique_analysis", critique_node)
    workflow.add_node("correct_analysis", correction_node)
    workflow.add_node("human_review", human_review_node)
    
    workflow.add_edge(START, "retrieve_data")
    workflow.add_edge("retrieve_data", "generate_draft")
    workflow.add_edge("generate_draft", "critique_analysis")
    
    workflow.add_conditional_edges(
        "critique_analysis",
        should_continue,
        {
            "correct_analysis": "correct_analysis",
            "human_review": "human_review",
            "END": END
        }
    )
    
    workflow.add_edge("correct_analysis", "critique_analysis")
    workflow.add_edge("human_review", END)
    
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

cyclical_reasoning_app = build_cyclical_reasoning_graph()
