import asyncio
import logging
import pandas as pd
from core.agents.algo_trading_agent import AlgoTradingAgent
from core.agents.supply_chain_risk_agent import SupplyChainRiskAgent
from core.agents.legal_agent import LegalAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
from core.agents.data_visualization_agent import DataVisualizationAgent
from core.agents.prediction_market_agent import PredictionMarketAgent
from core.agents.lexica_agent import LexicaAgent
from core.agents.lingua_maestro import LinguaMaestro
from core.agents.sense_weaver import SenseWeaver
from core.agents.natural_language_generation_agent import NaturalLanguageGenerationAgent
from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialistAgent
from core.agents.machine_learning_model_training_agent import MachineLearningModelTrainingAgent
from core.agents.prompt_tuner import PromptTuner
from core.agents.meta_cognitive_agent import MetaCognitiveAgent
from core.agents.meta_agents.crisis_simulation_agent import CrisisSimulationMetaAgent
from core.schemas.crisis_simulation import CrisisSimulationInput

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_agents():
    print("--- Verifying AlgoTradingAgent ---")
    algo_agent = AlgoTradingAgent(config={'strategies': ['momentum'], 'initial_balance': 1000})
    market_data = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=100), 'Close': [100 + i for i in range(100)]})
    res_algo = await algo_agent.execute(data=market_data)
    print("Algo Result:", res_algo)

    print("\n--- Verifying SupplyChainRiskAgent ---")
    sc_agent = SupplyChainRiskAgent(config={'news_api_key': 'mock_key'})
    res_sc = await sc_agent.execute()
    print("Supply Chain Result Keys:", res_sc.keys())

    print("\n--- Verifying LegalAgent ---")
    legal_agent = LegalAgent(config={})
    res_legal = await legal_agent.execute(task="review_agreement", document_text="Standard agreement")
    print("Legal Result:", res_legal)

    print("\n--- Verifying IndustrySpecialistAgent ---")
    ind_agent = IndustrySpecialistAgent(config={'sector': 'technology'})
    res_ind = await ind_agent.execute(task="analyze_industry")
    print("Industry Result:", res_ind)

    print("\n--- Verifying DataVisualizationAgent ---")
    viz_agent = DataVisualizationAgent(config={})
    res_viz = await viz_agent.execute(visualization_type="chart", chart_type="bar", data={'x': [1], 'y': [2]})
    print("Viz Result:", res_viz)

    print("\n--- Verifying PredictionMarketAgent ---")
    pred_agent = PredictionMarketAgent(config={})
    res_pred = await pred_agent.execute(event={'type': 'company_stock'})
    print("Prediction Result Keys:", res_pred.keys())

    print("\n--- Verifying LexicaAgent ---")
    lex_agent = LexicaAgent(config={'api_keys': {'news_api_key': 'mock'}})
    res_lex = await lex_agent.execute(query="test")
    print("Lexica Result Keys:", res_lex.keys())

    print("\n--- Verifying LinguaMaestro ---")
    ling_agent = LinguaMaestro(config={})
    res_ling = await ling_agent.execute(task="translate", text="Hello")
    print("Lingua Result:", res_ling)

    print("\n--- Verifying SenseWeaver ---")
    sense_agent = SenseWeaver(config={})
    res_sense = await sense_agent.execute(task="process_input", data={"raw": "test"})
    print("Sense Result:", res_sense)

    print("\n--- Verifying NLG Agent ---")
    nlg_agent = NaturalLanguageGenerationAgent(config={'model_name': 'distilgpt2'})
    res_nlg = await nlg_agent.execute(output_type="summary", data="Test data")
    print("NLG Result:", res_nlg)

    print("\n--- Verifying NewsletterLayoutSpecialistAgent ---")
    news_agent = NewsletterLayoutSpecialistAgent(config={})
    res_news = await news_agent.execute(data={'market_sentiment': {'summary': 'Good'}})
    print("Newsletter Result (Preview):", res_news[:50])

    print("\n--- Verifying MLTrainingAgent ---")
    ml_agent = MachineLearningModelTrainingAgent(config={})
    res_ml = await ml_agent.execute(data_sources=["mock"], model_type="linear_regression", model_name="test_model")
    print("ML Result:", res_ml)

    print("\n--- Verifying PromptTuner ---")
    tuner_agent = PromptTuner(config={})
    res_tuner = await tuner_agent.execute(task="analyze", prompt="Analyze AAPL")
    print("Tuner Result:", res_tuner)

    print("\n--- Verifying MetaCognitiveAgent ---")
    meta_agent = MetaCognitiveAgent(config={})
    content = "The expert says this is a strong buy but also highly risky."
    res_meta = await meta_agent.execute(content_to_analyze=content, agent_name="TestAgent")
    print("MetaCognitive Result:", res_meta)

    print("\n--- Verifying CrisisSimulationAgent ---")
    crisis_agent = CrisisSimulationMetaAgent(config={})

    # We will just verify it initializes without crashing, since full execute requires SK or Graph
    print("CrisisSimulationAgent Initialized:", crisis_agent is not None)

if __name__ == "__main__":
    asyncio.run(test_agents())
