import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

from core.agents.specialized.snc_rating_agent import SNCRatingAgent
from core.agents.specialized.covenant_analyst_agent import CovenantAnalystAgent
from core.schemas.v23_5_schema import CreditAnalysis

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_entity(name: str, financials: Dict, capital_structure: List[Dict], ev: float, covenant_threshold: float, snc_agent, covenant_agent):
    covenant_params = {
        "leverage": financials["total_debt"] / financials["ebitda"],
        "covenant_threshold": covenant_threshold
    }

    # EXECUTE WITH AWAIT
    # AgentBase wraps execute in an async wrapper (wrapped_execute), so we must await it.
    snc_result = await snc_agent.execute(financials=financials, capital_structure=capital_structure, enterprise_value=ev)
    covenant_result = await covenant_agent.execute(params=covenant_params)

    # Simulate CDS rating based on SNC rating
    cds_map = {
        "Pass": "BBB- (Spread: 150bps)",
        "Special Mention": "B+ (Spread: 450bps)",
        "Substandard": "CCC+ (Spread: 850bps)",
        "Doubtful": "D (Distressed)"
    }
    cds_rating = cds_map.get(snc_result.overall_borrower_rating, "NR")

    return {
        "entity_name": name,
        "analysis": CreditAnalysis(
            snc_rating_model=snc_result,
            cds_market_implied_rating=cds_rating,
            covenant_risk_analysis=covenant_result
        ).model_dump(mode='json')
    }

async def run_phase3_demo():
    logger.info("Starting Adam v23.5 Phase 3 Demo: The Debt Lens")

    config = {"mode": "demo", "use_v23_graph": False}
    snc_agent = SNCRatingAgent(config)
    covenant_agent = CovenantAnalystAgent(config)

    # Define Portfolio Scenarios
    scenarios = [
        {
            "name": "TechCo Inc. (Distressed)",
            "financials": {"ebitda": 100.0, "total_debt": 650.0, "interest_expense": 80.0, "free_cash_flow": 20.0},
            "cap_structure": [{"name": "Term Loan B", "amount": 650.0, "priority": 1, "security": "First Lien"}],
            "ev": 600.0,
            "covenant": 6.0
        },
        {
            "name": "SafeHarbor Logistics",
            "financials": {"ebitda": 200.0, "total_debt": 600.0, "interest_expense": 40.0, "free_cash_flow": 120.0},
            "cap_structure": [{"name": "Revolver", "amount": 100.0}, {"name": "Notes", "amount": 500.0}],
            "ev": 1800.0,
            "covenant": 5.5
        },
        {
            "name": "EdgeCase Retail",
            "financials": {"ebitda": 50.0, "total_debt": 240.0, "interest_expense": 25.0, "free_cash_flow": 10.0},
            "cap_structure": [{"name": "Term Loan", "amount": 240.0}],
            "ev": 260.0,
            "covenant": 5.0
        },
        {
             "name": "Apollo Portfolio Co A",
             "financials": {"ebitda": 150.0, "total_debt": 450.0, "interest_expense": 50.0, "free_cash_flow": 80.0},
             "cap_structure": [{"name": "Unitranche", "amount": 450.0}],
             "ev": 1200.0,
             "covenant": 6.5
        },
        {
             "name": "Apollo Portfolio Co B (Watchlist)",
             "financials": {"ebitda": 80.0, "total_debt": 480.0, "interest_expense": 60.0, "free_cash_flow": 5.0},
             "cap_structure": [{"name": "Term Loan", "amount": 480.0}],
             "ev": 500.0,
             "covenant": 6.25
        }
    ]

    results = []
    for s in scenarios:
        logger.info(f"Analyzing {s['name']}...")
        res = await analyze_entity(s['name'], s['financials'], s['cap_structure'], s['ev'], s['covenant'], snc_agent, covenant_agent)
        results.append(res)

    # Output Generation
    output_dir = "showcase/data"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Full Portfolio for Visualization
    portfolio_file = os.path.join(output_dir, "phase3_portfolio_demo.json")
    with open(portfolio_file, "w") as f:
        json.dump({"portfolio": results, "generated_at": datetime.now().isoformat()}, f, indent=2)

    # 2. Single Entity for Article Artifact (TechCo)
    techco_file = os.path.join(output_dir, "phase3_techco_artifact.json")
    with open(techco_file, "w") as f:
        # Wrap in HDKG-like structure for the "Article"
        artifact = {
            "v23_knowledge_graph": {
                "meta": {
                    "target": "TechCo Inc.",
                    "generated_at": datetime.now().isoformat(),
                    "model_version": "Adam-v23.5-Apex-Architect"
                },
                "nodes": {
                    "credit_analysis": results[0]['analysis']
                    # Other nodes omitted for this vertical slice
                }
            }
        }
        json.dump(artifact, f, indent=2)

    logger.info(f"Demo Complete. Data saved to {portfolio_file} and {techco_file}")

if __name__ == "__main__":
    asyncio.run(run_phase3_demo())
