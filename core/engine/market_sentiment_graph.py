# core/engine/market_sentiment_graph.py

"""
Agent Notes (Meta-Commentary):
This module implements the Market Sentiment & News Monitoring Graph.
It simulates an agent that "reads" news, calculates sentiment scores,
cross-references them with the Knowledge Graph (KG) to find contagion risks,
and issues alerts.

Architecture:
- Cyclic Graph: Uses LangGraph for the feedback loop (Analysis -> KG Check -> Draft).
- State: Managed via `MarketSentimentState`.
"""

import logging
import random
from typing import Any, Dict, List, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.engine.states import MarketSentimentState

logger = logging.getLogger(__name__)

# --- Mock Data Generators ---

def _mock_fetch_news(ticker: str) -> List[Dict[str, Any]]:
    """Simulates fetching news from an API."""
    templates = [
        {"title": f"{ticker} announces record profits", "sentiment": 0.8, "driver": "Earnings"},
        {"title": f"Supply chain issues hit {ticker}", "sentiment": -0.5, "driver": "Supply Chain"},
        {"title": f"CEO of {ticker} steps down", "sentiment": -0.3, "driver": "Governance"},
        {"title": f"Analysts upgrade {ticker} to Buy", "sentiment": 0.6, "driver": "Analyst Ratings"},
        {"title": f"Regulatory probe into {ticker} pricing", "sentiment": -0.9, "driver": "Regulation"},
    ]
    # Return a random subset
    return random.sample(templates, k=random.randint(1, 3))

# --- Nodes ---

def ingest_news_node(state: MarketSentimentState) -> Dict[str, Any]:
    """
    Node: Ingest News
    Fetches latest headlines and social signals.
    """
    print("--- Node: Ingest News ---")
    ticker = state["ticker"]

    # Simulate fetch
    articles = _mock_fetch_news(ticker)

    return {
        "news_feed": state["news_feed"] + articles,
        "human_readable_status": f"Ingested {len(articles)} new articles."
    }

def analyze_sentiment_node(state: MarketSentimentState) -> Dict[str, Any]:
    """
    Node: Analyze Sentiment
    Calculates aggregate score and identifies drivers.
    """
    print("--- Node: Analyze Sentiment ---")
    feed = state["news_feed"]

    if not feed:
        return {"sentiment_score": 0.0, "human_readable_status": "No news to analyze."}

    total_score = sum(a["sentiment"] for a in feed)
    avg_score = total_score / len(feed)

    # Determine Trend
    if avg_score > 0.3: trend = "bullish"
    elif avg_score < -0.3: trend = "bearish"
    else: trend = "neutral"

    # Extract drivers
    drivers = list(set(a["driver"] for a in feed))

    return {
        "sentiment_score": avg_score,
        "sentiment_trend": trend,
        "key_drivers": drivers,
        "human_readable_status": f"Sentiment is {trend} ({avg_score:.2f})."
    }

def kg_cross_reference_node(state: MarketSentimentState) -> Dict[str, Any]:
    """
    Node: KG Cross Reference
    Checks the Unified Knowledge Graph for related entities that might be affected.
    (e.g., If Apple has supply chain issues, check 'TechnologySector').
    """
    print("--- Node: KG Cross Reference ---")
    drivers = state["key_drivers"]
    sector = state["target_sector"]

    related = []

    # Mock Logic: In a real system, we'd query UnifiedKnowledgeGraph
    if "Supply Chain" in drivers:
        related.append(f"Suppliers of {state['ticker']}")
    if "Regulation" in drivers:
        related.append(f"Competitors in {sector}")
    if "Earnings" in drivers:
        related.append(f"{sector} ETF")

    return {
        "related_entities": related,
        "human_readable_status": f"Identified {len(related)} related entities in KG."
    }

def draft_alert_node(state: MarketSentimentState) -> Dict[str, Any]:
    """
    Node: Draft Alert
    Synthesizes findings into a report and sets alert level.
    """
    print("--- Node: Draft Alert ---")
    score = state["sentiment_score"]
    trend = state["sentiment_trend"]
    drivers = ", ".join(state["key_drivers"])
    related = ", ".join(state["related_entities"])

    # Determine Alert Level
    if score < -0.6 or score > 0.8:
        level = "HIGH"
    elif abs(score) > 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    report = f"MARKET ALERT: {state['ticker']}\n"
    report += f"Level: {level}\n"
    report += f"Trend: {trend.upper()} (Score: {score:.2f})\n"
    report += f"Drivers: {drivers}\n"
    if related:
        report += f"Contagion Risks: {related}\n"

    return {
        "alert_level": level,
        "final_report": report,
        "iteration_count": state["iteration_count"] + 1,
        "human_readable_status": f"Drafted {level} priority alert."
    }

# --- Conditional Logic ---

def should_continue(state: MarketSentimentState) -> Literal["ingest_news", "END"]:
    # Mock Loop: If we have "HIGH" alert but haven't checked 2 iterations, check news again for updates
    if state["alert_level"] == "HIGH" and state["iteration_count"] < 2:
        return "ingest_news"
    return "END"

# --- Graph Construction ---

def build_sentiment_graph():
    workflow = StateGraph(MarketSentimentState)

    workflow.add_node("ingest_news", ingest_news_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("kg_cross_reference", kg_cross_reference_node)
    workflow.add_node("draft_alert", draft_alert_node)

    workflow.add_edge(START, "ingest_news")
    workflow.add_edge("ingest_news", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "kg_cross_reference")
    workflow.add_edge("kg_cross_reference", "draft_alert")

    workflow.add_conditional_edges(
        "draft_alert",
        should_continue,
        {
            "ingest_news": "ingest_news",
            "END": END
        }
    )

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

sentiment_graph_app = build_sentiment_graph()
