import asyncio
import json
import logging
import random
import os
from typing import Dict, Any

from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.data_processing.synthetic_data_factory import DataFactory
from core.utils.logging_utils import SwarmLogger

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock Data Generation Helper
def generate_mock_company_data(ticker: str, scenario: str) -> Dict[str, Any]:
    # Reuse DataFactory logic for consistent scenario generation
    report = DataFactory.generate_deep_dive(ticker, scenario)

    # Transform report structure to what FundamentalAnalystAgent expects
    kg = report["v23_knowledge_graph"]["nodes"]
    financials = kg["equity_analysis"]["fundamentals"]
    valuation = kg["equity_analysis"]["valuation_engine"]

    # Synthetic Income Statement
    revenue = 1000000000 if ticker == "AAPL" else 500000000
    ebitda = revenue * (0.3 if scenario == "bull" else 0.15)
    net_income = ebitda * 0.6

    # Synthetic Balance Sheet
    assets = revenue * 2
    liabilities = assets * 0.4
    equity = assets - liabilities

    return {
        "company_info": {"name": f"{ticker} Inc", "industry_sector": "Tech", "country": "USA"},
        "financial_data_detailed": {
            "income_statement": {
                "revenue": [revenue * 0.8, revenue * 0.9, revenue],
                "net_income": [net_income * 0.8, net_income * 0.9, net_income],
                "ebitda": [ebitda * 0.8, ebitda * 0.9, ebitda]
            },
            "balance_sheet": {
                "total_assets": [assets, assets, assets],
                "total_liabilities": [liabilities, liabilities, liabilities],
                "shareholders_equity": [equity, equity, equity],
                "cash_and_equivalents": [assets * 0.1, assets * 0.1, assets * 0.1],
                "short_term_debt": [liabilities * 0.1],
                "long_term_debt": [liabilities * 0.4]
            },
            "cash_flow_statement": {
                "operating_cash_flow": [ebitda * 0.8],
                "investing_cash_flow": [-ebitda * 0.2],
                "financing_cash_flow": [-ebitda * 0.1],
                "free_cash_flow": [ebitda * 0.5]
            },
            "dcf_assumptions": {
                "fcf_projection_years_total": 10,
                "initial_high_growth_period_years": 5,
                "initial_high_growth_rate": 0.10 if scenario == "bull" else 0.02,
                "stable_growth_rate": 0.05,
                "discount_rate": 0.09,
                "terminal_growth_rate_perpetuity": 0.025
            },
            "market_data": {
                "share_price": valuation["dcf_model"]["intrinsic_share_price"],
                "shares_outstanding": 1000000
            }
        },
        "qualitative_company_info": {
            "management_assessment": kg["entity_ecosystem"]["management_assessment"],
            "competitive_advantages": kg["entity_ecosystem"]["competitive_positioning"]
        }
    }

async def run_scenario(ticker: str, scenario: str):
    logging.info(f"Running scenario: {ticker} ({scenario})")

    # Patch retrieve_company_data to return mock data
    mock_data = generate_mock_company_data(ticker, scenario)

    class MockFundamentalAnalystAgent(FundamentalAnalystAgent):
        async def retrieve_company_data(self, company_id: str) -> Dict[str, Any]:
             return mock_data

    agent = MockFundamentalAnalystAgent(config={"agent_id": "FundamentalAnalyst", "persona": "Senior Analyst"})

    try:
        result = await agent.execute(ticker)
        logging.info(f"Analysis complete for {ticker}")
    except Exception as e:
        logging.error(f"Analysis failed for {ticker}: {e}")

async def main():
    # Clear previous logs to avoid duplication if running multiple times in session
    log_file = "logs/swarm_telemetry.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)

    scenarios = [
        ("AAPL", "bull"),
        ("AAPL", "bear"),
        ("MSFT", "neutral"),
        ("GOOGL", "bull"),
        ("AMZN", "bear")
    ]

    for ticker, scenario in scenarios:
        await run_scenario(ticker, scenario)

    logging.info("All scenarios completed.")

    output_file = "data/artisanal_training_sets/runtime_captured_data.jsonl"

    if os.path.exists(log_file):
        logging.info(f"Processing logs from {log_file} to {output_file}")
        with open(log_file, "r") as f:
            events = [json.loads(line) for line in f]

        training_data = []
        pending_tasks = []

        for event in events:
            if event["event_type"] == "TASK_START":
                pending_tasks.append(event)
            elif event["event_type"] == "TASK_COMPLETE":
                if pending_tasks:
                    start_event = pending_tasks.pop()

                    details = start_event.get("details", {})
                    # args is a sibling of inputs in details
                    args = details.get("args", [])

                    outputs = event.get("details", {}).get("output", {})

                    # Create a readable prompt
                    company = "Unknown"
                    if args and len(args) > 0:
                        company = args[0]
                        # Clean up quotes if string repr added them
                        if company.startswith("'") and company.endswith("'"):
                            company = company[1:-1]

                    prompt = f"Perform a fundamental analysis for {company}."

                    completion = outputs.get("analysis_summary", str(outputs))

                    training_data.append({
                        "prompt": prompt,
                        "completion": completion
                    })

        with open(output_file, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        logging.info(f"Generated {len(training_data)} training examples.")
    else:
        logging.error("Log file not found!")

if __name__ == "__main__":
    asyncio.run(main())
