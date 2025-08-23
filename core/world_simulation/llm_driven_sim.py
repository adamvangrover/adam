# core/world_simulation/llm_driven_sim.py

import json
from typing import Dict, Any
from core.llm.base_llm_engine import BaseLLMEngine
from core.world_simulation.config import WorldSimulationConfig

class LLMDrivenSim:
    def __init__(self, config: WorldSimulationConfig, llm_engine: BaseLLMEngine):
        self.config = config
        self.llm_engine = llm_engine
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        with open("prompts/JSON_Prompt_Library.jsonl", "r") as f:
            for line in f:
                prompt_data = json.loads(line)
                if prompt_data["task_id"] == "WMSIM01":
                    return prompt_data["prompt_text"]
        raise ValueError("Could not find prompt with task_id WMSIM01")

    def _get_initial_state(self) -> Dict[str, Any]:
        return {
            "market": {
                "stock_prices": {symbol: 100.0 for symbol in self.config.market.stock_symbols},
                "volatility": self.config.market.volatility,
                "risk_aversion": self.config.market.risk_aversion,
                "market_sentiment": self.config.market.market_sentiment,
                "liquidity": self.config.market.liquidity,
                "news_impact": self.config.market.news_impact,
            },
            "economy": {
                "gdp_growth": self.config.economy.gdp_growth,
                "inflation": self.config.economy.inflation,
                "interest_rate": self.config.economy.interest_rate,
                "unemployment": self.config.economy.unemployment,
                "consumer_confidence": self.config.economy.consumer_confidence,
                "business_confidence": self.config.economy.business_confidence,
                "housing_starts": self.config.economy.housing_starts,
                "retail_sales": self.config.economy.retail_sales,
                "cpi": self.config.economy.cpi,
            },
            "geopolitics": {
                "political_stability": self.config.geopolitics.political_stability,
                "trade_war_risk": self.config.geopolitics.trade_war_risk,
                "regulatory_changes": self.config.geopolitics.regulatory_changes,
                "election_risk": self.config.geopolitics.election_risk,
                "geopolitical_hotspots": self.config.geopolitics.geopolitical_hotspots,
                "terrorism_risk": self.config.geopolitics.terrorism_risk,
            },
            "environment": {
                "natural_disaster_risk": self.config.environment.natural_disaster_risk,
                "climate_change_impact": self.config.environment.climate_change_impact,
            },
            "demographics": {
                "population_growth": self.config.demographics.population_growth,
                "aging_population_impact": self.config.demographics.aging_population_impact,
            },
            "technology": {
                "technological_disruption_risk": self.config.technology.technological_disruption_risk,
                "ai_adoption_rate": self.config.technology.ai_adoption_rate,
            },
        }

    def run_simulation(self) -> Dict[str, Any]:
        current_state = self._get_initial_state()
        history = []

        for _ in range(self.config.simulation.steps):
            prompt = self.prompt.format(current_state=json.dumps(current_state, indent=2))
            response = self.llm_engine.generate(prompt)
            
            try:
                response_json = json.loads(response)
                next_state = response_json["next_state"]
                condition_matrix = response_json["condition_matrix"]

                history.append(
                    {
                        "state": current_state,
                        "condition_matrix": condition_matrix,
                    }
                )
                current_state = next_state
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing LLM response: {e}")
                print(f"Response: {response}")
                # Decide how to handle the error: e.g., retry, use last known state, or stop.
                # For now, we'll just stop the simulation for this run.
                break


        return history
