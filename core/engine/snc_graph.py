# core/engine/snc_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements a comprehensive Examination Team Graph for Shared National Credits (SNC).
It orchestrates the analysis of complex syndicated loans by stringing together specialized agents:
- Lead Examiner (File Review)
- Regulatory Specialist (Compliance)
- Senior Credit Officer (Debate & Consensus)
- Documentarian (Automated Rating Report Generation)
"""

import logging
from typing import Dict, Any
from core.utils.graph_utils import StateGraph, END, START, MemorySaver, HAS_LANGGRAPH
from core.engine.states import SNCAnalysisState
from core.agents.specialized.regulatory_snc_agent import RegulatorySNCAgent
from core.engine.risk_consensus_engine import RiskConsensusEngine
from core.compliance.snc_validators import evaluate_compliance
from core.engine.snc_utils import analyze_syndicate_structure

logger = logging.getLogger(__name__)

# Initialize Agents/Engines for the nodes
dummy_config = {"name": "GraphTeam"}
regulator_agent = RegulatorySNCAgent(dummy_config)
consensus_engine = RiskConsensusEngine()


# --- Nodes ---


async def industry_and_historical_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Industry & Historical Trends
    Evaluates sector heuristics and historical performance against compliance guardrails.
    """
    print("--- Analyst: Evaluating Sector Heuristics & Historicals ---")

    sector = state.get("industry_sector", "General")
    market_data = state.get("market_data", {})
    hist_fin = state.get("historical_financials", {})
    fin = state.get("financials", {})

    # Calculate some key ratios needed by the validator
    ebitda = float(fin.get("ebitda", 0))
    debt = float(fin.get("total_debt", 0))
    interest = float(fin.get("interest_expense", 1))
    revenue = float(fin.get("revenue", 1))

    current_assets = float(fin.get("current_assets", 1))
    current_liabilities = float(fin.get("current_liabilities", 1))

    ratios = {
        "debt_to_equity_ratio": debt / (ev - debt) if (ev := state.get("enterprise_value", 0)) > debt else debt,
        "net_profit_margin": ebitda / revenue if revenue > 0 else 0,
        "current_ratio": current_assets / current_liabilities if current_liabilities > 0 else 1,
        "interest_coverage_ratio": ebitda / interest if interest > 0 else 0
    }

    financial_data = {"key_ratios": ratios}

    compliance_result = evaluate_compliance(financial_data, sector, market_data)

    analysis = f"Sector: {sector}. "
    if compliance_result.passed:
        analysis += "Sector heuristics and risk limits passed. "
    else:
        analysis += "Compliance Flags: " + ", ".join(compliance_result.violations) + ". "

    trend = hist_fin.get("revenue_trend", "Stable")
    analysis += f"Historical revenue trend is considered {trend}."

    return {
        "sector_heuristics_analysis": analysis,
        "human_readable_status": "Evaluated industry heuristics."
    }


async def file_review_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Lead Examiner Review
    Deep dive into financials and file data to determine structural risks and repayment capacity.
    """
    print("--- Examiner: Performing File Review ---")
    fin = state["financials"]
    ebitda = float(fin.get("ebitda", 0))
    debt = float(fin.get("total_debt", 0))
    interest = float(fin.get("interest_expense", 1))

    dscr = (ebitda) / interest if interest > 0 else 0
    leverage = debt / ebitda if ebitda > 0 else 99

    analyst_rationale = f"File Review Complete. Calculated core metrics: Leverage={leverage:.2f}x, DSCR={dscr:.2f}x. "

    if dscr < 1.0:
        analyst_rationale += "Critical warning: Cash flow does not cover interest. "
        analyst_rating = "Doubtful"
    elif leverage > 6.0:
        analyst_rationale += "Leverage is exceptionally high, indicating significant structural risk. "
        analyst_rating = "Substandard"
    elif dscr < 1.2 or leverage > 4.0:
        analyst_rationale += "Tight coverage requires monitoring. "
        analyst_rating = "Special Mention"
    else:
        analyst_rationale += "Metrics appear within acceptable historical bounds. "
        analyst_rating = "Pass"

    return {
        "analyst_rating": analyst_rating,
        "analyst_rationale": analyst_rationale,
        "human_readable_status": "Completed deep file review."
    }


async def syndicate_and_defense_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Syndicate Structure & Agent Defense
    Generates the narrative defense for the deal structure and syndication risk.
    """
    print("--- Syndicate: Generating Deal Defense ---")
    syndicate = state.get("syndicate_data", {})

    # Analyze structure
    struct_analysis = analyze_syndicate_structure(syndicate)

    # Generate Defense Narrative
    defense = "Agent Defense: The transaction structure is adequately supported. "
    if "Concentration Risk" in struct_analysis:
        defense += (
            "While lead bank share is high, it reflects strong conviction and skin-in-the-game, "
            "aligning interests with participants. "
        )
    elif "Leadership Risk" in struct_analysis:
        defense += (
            "The broad syndication mitigates individual institutional risk and "
            "demonstrates strong market appetite. "
        )
    else:
        defense += "The syndication is well-balanced across participating institutions. "

    fin = state.get("financials", {})
    if float(fin.get("liquidity", 0)) > float(fin.get("total_debt", 0)) * 0.1:
        defense += "Liquidity buffers are robust enough to handle short-term volatility."

    combined = struct_analysis + "\n" + defense

    return {
        "syndicate_and_defense": combined,
        "human_readable_status": "Prepared agent defense narrative."
    }


async def regulatory_review_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Regulatory Specialist Review
    Applies Interagency Guidance on Leveraged Lending.
    """
    print("--- Regulator: Applying Interagency Guidance ---")
    cap_struct = state.get("capital_structure", [])
    ev = state.get("enterprise_value", 0.0)

    result = await regulator_agent.execute(state["financials"], cap_struct, ev)

    # 2013 Interagency Guidance Specific Watch Points
    watch_points = "Regulator Watch Points:\n"
    if result.overall_borrower_rating in ["Substandard", "Doubtful", "Loss"]:
        watch_points += "- [ALERT] High leverage exceeds 6.0x guidance limits. "
        watch_points += "- De-leveraging capacity within 5-7 years is questionable without material EBITDA growth. "

    fin = state.get("financials", {})
    debt = float(fin.get("total_debt", 0))
    if ev > 0 and (debt / ev) > 0.75:
        watch_points += (
            "- Enterprise Value reliance is extremely high (LTV > 75%), limiting secondary repayment sources. "
        )

    if not watch_points.endswith(" "):
        watch_points += "- No major structural triggers outside standard covenant monitoring identified."

    return {
        "regulatory_rating": result.overall_borrower_rating,
        "regulatory_rationale": result.rationale,
        "regulator_watch_points": watch_points,
        "human_readable_status": "Completed regulatory compliance check."
    }


async def committee_debate_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Senior Credit Officer
    Orchestrates the debate between the Regulatory and Analyst views to reach consensus.
    """
    print("--- Committee: Orchestrating Debate & Consensus ---")
    reg_rating = state["regulatory_rating"]
    reg_rat = state["regulatory_rationale"]

    strat_rating = state["analyst_rating"]
    strat_rat = state["analyst_rationale"]
    strat_confidence = 0.85  # Assume high confidence from deep fundamental review

    result = consensus_engine.calculate_consensus(
        reg_rating=reg_rating,
        strat_rating=strat_rating,
        strat_confidence=strat_confidence,
        reg_rationale=reg_rat,
        strat_rationale=strat_rat
    )

    return {
        "consensus_rating": result.final_rating,
        "consensus_rationale": result.narrative,
        "risk_dialogue": result.risk_dialogue.model_dump() if result.risk_dialogue else None,
        "human_readable_status": "Consensus reached via debate."
    }


async def report_generation_node(state: SNCAnalysisState) -> Dict[str, Any]:
    """
    Node: Documentarian
    Compiles the findings, debate, and final rating into an automated report.
    """
    print("--- Documentarian: Generating Automated Rating Report ---")
    dialogue_text = ""
    if state.get("risk_dialogue"):
        for turn in state["risk_dialogue"].get("turns", []):
            dialogue_text += f"**{turn['speaker']}**: {turn['argument']}\n\n"

    report = f"""# Shared National Credit (SNC) Examination Report
**Obligor ID:** {state.get('obligor_id', 'Unknown')}

## 1. Executive Summary
**Final Assigned Rating:** {state.get('consensus_rating')}
**Committee Rationale:** {state.get('consensus_rationale')}

## 2. Sector & Historical Context
{state.get('sector_heuristics_analysis')}

## 3. Syndicate Structure & Agent Defense
{state.get('syndicate_and_defense')}

## 4. Regulatory Watch Points
{state.get('regulator_watch_points')}

## 5. Examination Team Findings

### Lead Examiner (Fundamental Review)
*   **Proposed Rating:** {state.get('analyst_rating')}
*   **Analysis:** {state.get('analyst_rationale')}

### Regulatory Specialist (Compliance Review)
*   **Proposed Rating:** {state.get('regulatory_rating')}
*   **Analysis:** {state.get('regulatory_rationale')}

## 6. Committee Debate Transcript
{dialogue_text}

*Report automatically generated by SNC Agentic Examination Team.*
"""
    return {
        "final_report": report,
        "human_readable_status": "Final report generated."
    }


# --- Graph Construction ---


def build_snc_team_graph():
    if not HAS_LANGGRAPH:
        logger.warning("LangGraph not available. SNC Graph will be mocked.")

    workflow = StateGraph(SNCAnalysisState)

    workflow.add_node("file_review", file_review_node)
    workflow.add_node("industry_and_historical", industry_and_historical_node)
    workflow.add_node("syndicate_and_defense", syndicate_and_defense_node)
    workflow.add_node("regulatory_review", regulatory_review_node)
    workflow.add_node("committee_debate", committee_debate_node)
    workflow.add_node("report_generation", report_generation_node)

    workflow.add_edge(START, "file_review")
    workflow.add_edge("file_review", "industry_and_historical")
    workflow.add_edge("industry_and_historical", "syndicate_and_defense")
    workflow.add_edge("syndicate_and_defense", "regulatory_review")
    workflow.add_edge("regulatory_review", "committee_debate")
    workflow.add_edge("committee_debate", "report_generation")
    workflow.add_edge("report_generation", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


snc_graph_app = build_snc_team_graph()
