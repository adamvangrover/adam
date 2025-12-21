# core/world_simulation/config.py

from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class LLMConfig:
    engine: str = "openai"
    model: str = "gpt-4-turbo"
    temperature: float = 0.5
    max_tokens: int = 2048

@dataclass
class MarketConfig:
    volatility: float = 0.08
    risk_aversion: float = 0.6
    stock_symbols: List[str] = field(default_factory=list)
    market_sentiment: float = 0.5
    liquidity: float = 0.7
    news_impact: float = 0.3

@dataclass
class EconomyConfig:
    gdp_growth: float = 0.025
    inflation: float = 0.032
    interest_rate: float = 0.055
    unemployment: float = 0.045
    consumer_confidence: float = 0.6
    business_confidence: float = 0.7
    housing_starts: float = 1.5
    retail_sales: float = 0.01
    cpi: float = 250.0

@dataclass
class GeopoliticsConfig:
    political_stability: float = 0.75
    trade_war_risk: float = 0.15
    regulatory_changes: float = 0.12
    election_risk: float = 0.2
    geopolitical_hotspots: List[str] = field(default_factory=list)
    terrorism_risk: float = 0.1

@dataclass
class EnvironmentConfig:
    natural_disaster_risk: float = 0.05
    climate_change_impact: float = 0.02

@dataclass
class DemographicsConfig:
    population_growth: float = 0.008
    aging_population_impact: float = 0.01

@dataclass
class TechnologyConfig:
    technological_disruption_risk: float = 0.1
    ai_adoption_rate: float = 0.05

@dataclass
class SimulationConfig:
    steps: int = 200
    runs: int = 50

@dataclass
class WorldSimulationConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    geopolitics: GeopoliticsConfig = field(default_factory=GeopoliticsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
    technology: TechnologyConfig = field(default_factory=TechnologyConfig)

def load_config(path: str) -> WorldSimulationConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return WorldSimulationConfig(
        simulation=SimulationConfig(**config_dict.get("simulation", {})),
        llm=LLMConfig(**config_dict.get("llm", {})),
        market=MarketConfig(**config_dict.get("market", {})),
        economy=EconomyConfig(**config_dict.get("economy", {})),
        geopolitics=GeopoliticsConfig(**config_dict.get("geopolitics", {})),
        environment=EnvironmentConfig(**config_dict.get("environment", {})),
        demographics=DemographicsConfig(**config_dict.get("demographics", {})),
        technology=TechnologyConfig(**config_dict.get("technology", {})),
    )
