
import asyncio
import logging
import sys
import os
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.algo_trading_agent import AlgoTradingAgent
from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.agents.snc_analyst_agent import SNCAnalystAgent
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.specialized.defi_liquidity_agent import DeFiLiquidityAgent
from core.agents.specialized.crypto_arbitrage_agent import CryptoArbitrageAgent

from core.agents.behavioral_economics_agent import BehavioralEconomicsAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.anomaly_detection_agent import AnomalyDetectionAgent

# New Batch
from core.agents.legal_agent import LegalAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.supply_chain_risk_agent import SupplyChainRiskAgent

from core.schemas.agent_schema import AgentInput, AgentOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerificationScript")

async def verify_agent(agent_name, agent_instance, input_data):
    logger.info(f"--- Verifying {agent_name} ---")
    try:
        # Test Standard Execution (AgentInput)
        logger.info(f"Testing execute() with AgentInput...")
        result = await agent_instance.execute(input_data)

        if isinstance(result, AgentOutput):
            logger.info(f"PASS: Returned AgentOutput.")
            logger.info(f"Answer Preview: {result.answer[:100]}...")
            return True
        else:
            logger.warning(f"WARN: Returned {type(result)}, expected AgentOutput.")
            logger.error(f"FAIL: {agent_name} did not return AgentOutput when passed AgentInput.")
            return False

    except Exception as e:
        logger.error(f"FAIL: Exception during verification of {agent_name}: {e}", exc_info=True)
        return False

async def main():
    results = {}

    # Setup
    algo_config = {'strategies': ['momentum'], 'initial_balance': 10000}
    mock_data = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=100), 'Close': [100 + i + (i%5)*2 for i in range(100)]})
    standalone_input = AgentInput(query="SNC Manual Analysis", context={"manual_data": {"key_ratios": {"debt_to_equity_ratio": 5.0, "net_profit_margin": -0.10}}})

    # Execute Existing Group 1 & 2
    results['AlgoTradingAgent'] = await verify_agent('AlgoTradingAgent', AlgoTradingAgent(algo_config), AgentInput(query="Run Momentum", context={'data': mock_data.to_dict()}))
    results['RiskAssessmentAgent'] = await verify_agent('RiskAssessmentAgent', RiskAssessmentAgent({}), AgentInput(query="Assess Risk for Apple", context={'target_data': {'company_name': 'AAPL'}}))
    results['SNCAnalystAgent_Standalone'] = await verify_agent('SNCAnalystAgent', SNCAnalystAgent({}), standalone_input)
    results['BehavioralEconomicsAgent'] = await verify_agent('BehavioralEconomicsAgent', BehavioralEconomicsAgent({}), AgentInput(query="Behavior Analysis", context={"analysis_content": "panic dumping"}))
    results['GeopoliticalRiskAgent'] = await verify_agent('GeopoliticalRiskAgent', GeopoliticalRiskAgent({}), AgentInput(query="Assess US Risk", context={"regions": ["US"]}))
    results['AnomalyDetectionAgent'] = await verify_agent('AnomalyDetectionAgent', AnomalyDetectionAgent({}), AgentInput(query="Detect Anomalies"))

    # New Group 3
    results['LegalAgent'] = await verify_agent(
        'LegalAgent',
        LegalAgent({}),
        AgentInput(query="Review Agreement", context={"task": "review_agreement", "document_text": "The loan includes a negative pledge and a cross-default clause."})
    )

    results['MacroeconomicAnalysisAgent'] = await verify_agent(
        'MacroeconomicAnalysisAgent',
        MacroeconomicAnalysisAgent({}),
        AgentInput(query="2026 Reflation Outlook", context={"country": "US"})
    )

    results['SupplyChainRiskAgent'] = await verify_agent(
        'SupplyChainRiskAgent',
        SupplyChainRiskAgent({}),
        AgentInput(query="Supply Chain Disruption in Taiwan", context={"supplier_data": [{"name": "TSMC", "location": "Taiwan"}]})
    )


    print("\n=== Verification Summary ===")
    all_passed = True
    for agent, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{agent}: {status}")
        if not passed: all_passed = False

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
