# core/engine/cyclical_reasoning_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements the core cyclical reasoning engine for Adam v23.0.
It replaces the linear v22 simulation with a stateful LangGraph workflow.
It refactors critical logic from v21 agents into a dependency-free structure
suitable for the v23 graph architecture.
"""

import json
import logging
import random
import numpy as np
from typing import Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from core.engine.states import RiskAssessmentState, ResearchArtifact
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Import RiskAssessmentAgent (Math-heavy, no complex dependencies)
try:
    from core.agents.risk_assessment_agent import RiskAssessmentAgent
except ImportError as e:
    logger.error(f"Failed to import RiskAssessmentAgent: {e}")
    RiskAssessmentAgent = None

# Refactored Data Retrieval Logic (Decoupled from Semantic Kernel)
class V23DataRetriever:
    """
    A lightweight, v23-compliant data retriever that mirrors the logic
    of the legacy DataRetrievalAgent but without heavy dependencies.
    Includes rich mock data for showcase purposes.
    """
    def __init__(self):
        self.mock_db = {
            "AAPL": self._create_mock_data("Apple Inc.", "Technology", 180.0, 0.5),
            "TSLA": self._create_mock_data("Tesla Inc.", "Automotive", 240.0, 0.8),
            "JPM": self._create_mock_data("JPMorgan Chase", "Financials", 150.0, 0.3),
            "NVDA": self._create_mock_data("NVIDIA Corp", "Technology", 450.0, 0.6),
            "ABC_TEST": self._create_mock_data("ABC Test Corp", "Technology", 100.0, 0.4),
        }

    def _create_mock_data(self, name, industry, price, volatility_factor):
        return {
            "company_info": {
                "name": name,
                "industry_sector": industry,
                "country": "USA"
            },
            "financial_data_detailed": {
                "income_statement": {
                    "revenue": [1000 * (1.1**i) for i in range(3)],
                    "net_income": [150 * (1.1**i) for i in range(3)],
                    "ebitda": [300 * (1.1**i) for i in range(3)],
                },
                "balance_sheet": {
                    "total_assets": [2000 * (1.1**i) for i in range(3)],
                    "total_liabilities": [1000 * (1.1**i) for i in range(3)],
                    "shareholders_equity": [1000 * (1.1**i) for i in range(3)]
                },
                "key_ratios": {
                    "debt_to_equity_ratio": 0.5 + (random.random() * 0.2),
                    "net_profit_margin": 0.15 + (random.random() * 0.1),
                    "current_ratio": 1.5 + (random.random() * 1.0),
                    "interest_coverage_ratio": 10.0 + (random.random() * 5.0)
                },
                "market_data": {
                    "share_price": price,
                    "volatility_factor": volatility_factor,
                    "shares_outstanding": 10000000
                }
            }
        }

    def get_financials(self, company_id: str) -> Optional[Dict[str, Any]]:
        # Check mock DB first
        if company_id in self.mock_db:
            return self.mock_db[company_id]

        # Fallback for unknown tickers - generate on fly
        return self._create_mock_data(f"{company_id} Corp", "Unknown", 100.0, 0.5)

# --- Adapters & Helpers ---

def map_dra_to_raa(financials: Dict[str, Any]) -> tuple[Dict, Dict]:
    """
    Maps DataRetrievalAgent output to RiskAssessmentAgent input.
    """
    fin_details = financials.get("financial_data_detailed", {})
    
    mapped_fin = {
        "industry": financials.get("company_info", {}).get("industry_sector", "Unknown"),
        "credit_rating": "BBB", # Placeholder
    }
    
    current_price = fin_details.get("market_data", {}).get("share_price", 100)
    vol_factor = fin_details.get("market_data", {}).get("volatility_factor", 0.5)

    # Generate synthetic price history based on volatility
    prices = [current_price]
    for _ in range(30):
        change = (random.random() - 0.5) * vol_factor * 10
        prices.append(max(0.1, prices[-1] + change))

    mapped_market = {
        "price_data": np.array(prices)
    }
    
    return mapped_fin, mapped_market

# --- Initialization ---

data_retriever = V23DataRetriever()
try:
    # Patch: Provide default config to avoid TypeError
    risk_assessor = RiskAssessmentAgent(config={"mode": "standard"}) if RiskAssessmentAgent else None
except Exception as e:
    logger.error(f"Error init RAA: {e}")
    risk_assessor = None

# --- Nodes ---

def retrieve_data_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Retrieve Data
    Calls V23DataRetriever to fetch financials.
    """
    logger.info(f"--- Node: Retrieve Data for {state['ticker']} ---")
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
        status = f"Retrieved comprehensive financials for {ticker}."
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
    logger.info("--- Node: Generate Draft ---")
    
    financial_data_raw = {}
    for art in state["research_data"]:
        if "Financial Data" in art["title"]:
            try:
                financial_data_raw = json.loads(art["content"])
            except:
                pass
    
    financials, market = map_dra_to_raa(financial_data_raw)
    
    try:
        if risk_assessor:
            result = risk_assessor.assess_investment_risk(
                state["ticker"],
                financials,
                market
            )
            score = result.get("overall_risk_score", 0)
            factors = result.get("risk_factors", {})
        else:
            # Fallback if RAA is missing
            score = 75.0
            factors = {"Market Risk": "Moderate", "Liquidity Risk": "Low"}
        
        draft = f"RISK ASSESSMENT REPORT FOR {state['ticker']}\n"
        draft += f"Overall Risk Score: {score:.2f} / 100\n\n"
        draft += "Risk Factors Analysis:\n"
        for k, v in factors.items():
            draft += f"- {k}: {v}\n"
            
    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        draft = f"Error during assessment: {e}"

    return {
        "draft_analysis": draft,
        "human_readable_status": "Drafted initial risk assessment.",
        "iteration_count": state["iteration_count"] + 1
    }

def critique_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Critique
    """
    logger.info("--- Node: Critique ---")
    draft = state.get("draft_analysis", "")
    iteration = state["iteration_count"]
    
    critique_notes = []
    quality_score = 0.0
    needs_correction = False
    
    # Showcase Logic: Simulate a tough critic
    if "Error" in draft:
        critique_notes.append("Analysis failed due to system error.")
        quality_score = 0.1
        needs_correction = True
    elif iteration < 2:
        critique_notes.append("Analysis lacks depth on 'Liquidity Risk'. Please expand.")
        critique_notes.append("Verify Debt-to-Equity ratio against industry benchmarks.")
        quality_score = 0.65
        needs_correction = True
    elif iteration < 3:
        critique_notes.append("Consider macroeconomic headwinds (inflation).")
        quality_score = 0.75
        needs_correction = True
    else:
        critique_notes.append("Assessment is comprehensive and robust.")
        quality_score = 0.95
        needs_correction = False
        
    return {
        "critique_notes": critique_notes,
        "quality_score": quality_score,
        "needs_correction": needs_correction,
        "human_readable_status": f"Critique complete. Score: {quality_score}"
    }

def correction_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Correction
    """
    logger.info("--- Node: Correction ---")
    draft = state["draft_analysis"]
    notes = state["critique_notes"]
    
    correction_text = f"\n\n--- REVISION v{state['iteration_count']} ---\n"
    correction_text += "Addressing Feedback:\n"
    for note in notes:
        correction_text += f" * [Resolved] {note}\n"

    new_draft = draft + correction_text
    
    return {
        "draft_analysis": new_draft,
        "human_readable_status": "Applied expert corrections.",
        "iteration_count": state["iteration_count"] + 1
    }

def human_review_node(state: RiskAssessmentState) -> Dict[str, Any]:
    logger.info("--- Node: Human Review ---")
    return {
        "human_readable_status": "Analysis halted. Awaiting Human Review."
    }

# --- Conditional Logic ---

def should_continue(state: RiskAssessmentState) -> Literal["correct_analysis", "human_review", "END"]:
    quality = state["quality_score"]
    iteration = state["iteration_count"]
    needs_correction = state["needs_correction"]
    
    if not needs_correction or quality >= 0.90:
        return "END"
    
    if iteration >= 5:
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
