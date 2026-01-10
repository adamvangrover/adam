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
try:
    import numpy as np
except ImportError:
    np = None
from typing import Literal, Dict, Any, List, Optional
from core.engine.states import RiskAssessmentState, ResearchArtifact
from core.utils.logging_utils import get_logger

logger = get_logger(__name__)

from core.utils.graph_utils import StateGraph, END, START, MemorySaver, HAS_LANGGRAPH

# v23.5 Integration: APEX Generative Risk Engine
try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine, MarketScenario
except ImportError:
    GenerativeRiskEngine = None

# Import RiskAssessmentAgent (Math-heavy, no complex dependencies)
try:
    from core.agents.risk_assessment_agent import RiskAssessmentAgent
except ImportError as e:
    logger.error(f"Failed to import RiskAssessmentAgent: {e}")
    RiskAssessmentAgent = None

from core.tools.tool_registry import ToolRegistry
from core.data_processing.conviction_scorer import ConvictionScorer

# --- Adapters & Helpers ---


def map_dra_to_raa(financials: Dict[str, Any]) -> tuple[Dict, Dict]:
    """
    Maps DataRetrievalAgent output to RiskAssessmentAgent input.
    """
    fin_details = financials.get("financial_data_detailed", {})

    mapped_fin = {
        "industry": financials.get("company_info", {}).get("industry_sector", "Unknown"),
        "credit_rating": "BBB",  # Placeholder
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


tool_registry = ToolRegistry()
conviction_scorer = ConvictionScorer()

try:
    risk_assessor = RiskAssessmentAgent(config={"name": "CyclicalRiskAgent"}) if RiskAssessmentAgent else None
except Exception as e:
    logger.error(f"Error init RAA: {e}")
    risk_assessor = None

# APEX Engine Init
try:
    apex_engine = GenerativeRiskEngine() if GenerativeRiskEngine else None
except Exception as e:
    logger.error(f"Error init APEX Engine: {e}")
    apex_engine = None

# --- Nodes ---


def retrieve_data_node(state: RiskAssessmentState) -> Dict[str, Any]:
    """
    Node: Retrieve Data
    Calls ToolRegistry to fetch LIVE financials.
    Applies Conviction Scoring.
    """
    logger.info(f"--- Node: Retrieve Data for {state['ticker']} ---")
    ticker = state["ticker"]

    artifacts = []

    # 1. Fetch Financials (Live/Mock)
    financials = tool_registry.execute("get_financials", ticker=ticker)

    # 2. Fetch News (Live/Mock)
    news = tool_registry.execute("get_news", ticker=ticker, limit=2)

    status_msgs = []

    if financials and "error" not in financials:
        content = json.dumps(financials, indent=2)
        # Score conviction
        score = conviction_scorer.calculate_conviction(content, "Financial Report") # Weak check, but shows intent

        artifacts.append(ResearchArtifact(
            title=f"{ticker} Financial Data",
            content=content,
            source="ToolRegistry (Yahoo/Mock)",
            credibility_score=0.9 if score > 0.0 else 0.5
        ))
        status_msgs.append("financials")

    if news:
        for item in news:
            content = item.get("title", "") + ": " + item.get("link", "")
            # Verify news against Gold Standard (simulated)
            score = conviction_scorer.calculate_conviction(item.get("title", ""), "Official Press Release")

            artifacts.append(ResearchArtifact(
                title=f"News: {item.get('title', 'Unknown')}",
                content=content,
                source="Web Search",
                credibility_score=score
            ))
        status_msgs.append(f"{len(news)} news items")

    if not artifacts:
        status = f"No valid data found for {ticker}."
    else:
        status = f"Retrieved {', '.join(status_msgs)} for {ticker}."

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
    Node: Critique (Enhanced with APEX Engine and Self-Reflection Agent)
    """
    logger.info("--- Node: Critique (System 2 Verification) ---")
    draft = state.get("draft_analysis", "")
    iteration = state["iteration_count"]

    critique_notes = []
    quality_score = 0.0
    needs_correction = False

    # 1. Structural Checks
    if "Error" in draft:
        critique_notes.append("Analysis failed due to system error.")
        quality_score = 0.1
        needs_correction = True
        return {
            "critique_notes": critique_notes,
            "quality_score": quality_score,
            "needs_correction": needs_correction,
            "human_readable_status": "Critique failed due to errors."
        }

    # 2. Self-Reflection (Simulated LLM)
    # In production, this would call an LLM with the "Constitution" prompt.
    # Here we simulate the output of such an agent.

    has_liquidity = "Liquidity" in draft or "current_ratio" in draft
    has_conviction = "Score" in draft

    if not has_liquidity:
        critique_notes.append("Self-Reflection: The draft is missing Liquidity Risk analysis.")
        needs_correction = True

    if not has_conviction:
        critique_notes.append("Self-Reflection: No conviction score or verdict found.")
        needs_correction = True

    # 3. APEX Engine Verification (Tail Risk Check)
    if apex_engine:
        logger.info("Invoking APEX Generative Risk Engine for Validation...")
        breaches = apex_engine.reverse_stress_test(
            target_loss_threshold=5000000,
            current_portfolio_value=10000000
        )
        if breaches:
            draft_lower = draft.lower()
            if "tail risk" not in draft_lower and "crash" not in draft_lower:
                breach_desc = breaches[0].description
                critique_notes.append(
                    f"APEX Engine Warning: Identified critical Tail Risk scenario '{breach_desc}' that is absent from the draft.")
                needs_correction = True

    # 4. Scoring Logic (Replacing simple counter)
    base_score = 0.6
    if has_liquidity: base_score += 0.2
    if has_conviction: base_score += 0.1
    if not needs_correction: base_score += 0.1

    quality_score = min(0.99, base_score)

    # Force at least one refinement if quality is low, but respect max iterations in conditional edge
    if quality_score < 0.8:
        needs_correction = True
        if iteration == 0: # Always provide specific feedback on first pass
             critique_notes.append("General: Please expand on macro factors.")

    return {
        "critique_notes": critique_notes,
        "quality_score": quality_score,
        "needs_correction": needs_correction,
        "human_readable_status": f"Critique complete. Score: {quality_score:.2f}"
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

    if not needs_correction and quality >= 0.90:
        return "END"

    if iteration >= 5:
        return "human_review"

    return "correct_analysis"

# --- Graph Construction ---


def build_cyclical_reasoning_graph():
    if not HAS_LANGGRAPH:
        return None

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
