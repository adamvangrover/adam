# core/engine/data_fetcher.py

import json
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_gold_standard_kg() -> list:
    """Loads the v23.5 Knowledge Graph artifact."""
    try:
        # Strategy: Look for data directory relative to CWD or this file
        potential_paths = [
            "data/gold_standard/v23_5_knowledge_graph.json",
            "../../data/gold_standard/v23_5_knowledge_graph.json",
            "/app/data/gold_standard/v23_5_knowledge_graph.json"
        ]

        for p in potential_paths:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    return json.load(f)

        # Fallback: Try finding relative to this file
        rel_path = os.path.join(os.path.dirname(__file__), "../../data/gold_standard/v23_5_knowledge_graph.json")
        if os.path.exists(rel_path):
            with open(rel_path, 'r') as f:
                return json.load(f)

    except Exception as e:
        logger.warning(f"Failed to load Gold Standard KG: {e}")
    return []

def fetch_financial_context(ticker: str) -> Dict[str, Any]:
    """
    Fetches financial context from the Data Lakehouse (Gold Standard)
    or falls back to synthetic data for simulation.

    This implements the "Universal Ingestor" pattern where data is
    normalized into a standard schema for the Deep Dive Graph.
    """
    logger.info(f"Data Fetcher: Searching for {ticker}...")

    # 1. Try Gold Standard KG for Metadata/Pre-computed analysis
    kg_data = load_gold_standard_kg()
    entity_node = None

    # Normalize ticker
    search_term = ticker.lower()

    for entry in kg_data:
        target = entry.get("meta", {}).get("target", "").lower()
        name = entry.get("nodes", {}).get("entity_ecosystem", {}).get("legal_entity", {}).get("name", "").lower()

        if search_term in target or search_term in name:
            entity_node = entry
            break

    # 2. Construct Base Context (Synthetic/Default)
    # This ensures downstream math (DCF, etc.) always has valid inputs
    context = {
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
        ],
        "source": "Synthetic Baseline"
    }

    # 3. Enrich with Real Data if found
    if entity_node:
        logger.info(f"Data Fetcher: Found existing Knowledge Graph node for {ticker}. Enriching context.")
        context["source"] = "Hybrid (Gold Standard Metadata + Synthetic Financials)"
        context["kg_metadata"] = entity_node.get("meta")

        # Apply specific heuristics based on the known entity profile in the KG
        # This simulates "fetching" the real financials that led to the KG result

        target_name = entity_node.get("meta", {}).get("target", "")

        if "GameStop" in target_name:
            # Distressed profile
            context["fundamentals"]["ebitda"] = -150
            context["fundamentals"]["fcf"] = -300
            context["fundamentals"]["growth_rate"] = -0.05
            context["market_data"]["volatility"] = 0.80 # High vol
            context["syndicate_data"]["facilities"] = [{"id": "ABL", "amount": 500, "rate": "S+200"}]

        elif "Ford" in target_name:
            # Cyclical/Debt heavy
            context["fundamentals"]["total_debt"] = 140000 # High debt (Ford Credit)
            context["fundamentals"]["ebitda"] = 12000
            context["fundamentals"]["beta"] = 1.4

        elif "Apple" in target_name:
            # Cash rich
            context["fundamentals"]["cash_equivalents"] = 60000
            context["fundamentals"]["total_debt"] = 100000
            context["fundamentals"]["ebitda"] = 120000
            context["fundamentals"]["growth_rate"] = 0.08

    return context
