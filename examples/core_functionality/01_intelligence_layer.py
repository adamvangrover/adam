"""
Example 01: Intelligence Layer Standalone
-----------------------------------------
Demonstrates running the reasoning engine (Intelligence Layer) in isolation.
Uses the specialized RiskAssessmentAgent to evaluate investment risk without external dependencies.
"""

import sys
import os
import asyncio

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.system.provenance_logger import ProvenanceLogger, ActivityType

def run_intelligence_layer():
    print(">>> Initializing Intelligence Layer (RiskAssessmentAgent)...")

    # 1. Initialize Agent
    # Config can be minimal as we are mocking knowledge base access or using defaults
    config = {
        "knowledge_base_path": "data/risk_rating_mapping.json", # Can be dummy path
        "debug_mode": True
    }

    agent = RiskAssessmentAgent(config=config)
    logger = ProvenanceLogger()

    # 2. Define Input Data (Simulating a data packet from Data Layer)
    target_company = {
        "company_name": "TechGlobal Inc",
        "financial_data": {
            "credit_rating": "BBB",
            "cash": 5000000,
            "monthly_burn_rate": 200000,
            "total_assets": 12000000,
            "total_debt": 4000000
        },
        "market_data": {
            "trading_volume": 150000,
            "price_data": [100, 102, 101, 105, 103] # Simple price history
        }
    }

    print(f"[{agent.__class__.__name__}] Assessing: {target_company['company_name']}")

    # 3. Execute Reasoning (Async)
    # The agent encapsulates the business logic (System 2)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(agent.execute(
        target_data=target_company,
        risk_type="investment"
    ))

    print(f"[{agent.__class__.__name__}] Assessment Complete.")
    print(f"Overall Score: {result.get('overall_risk_score')}")
    print(f"Risk Factors: {result.get('risk_factors')}")

    # 4. Log Provenance
    logger.log_activity(
        agent_id="RiskAssessmentAgent",
        activity_type=ActivityType.DECISION,
        input_data={"company": target_company['company_name']},
        output_data=result,
        data_source="StandaloneExample",
        capture_full_io=True
    )

    print(">>> Provenance logs written to core/libraries_and_archives/audit_trails/")

if __name__ == "__main__":
    run_intelligence_layer()
