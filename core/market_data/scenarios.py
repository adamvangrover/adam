from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ScenarioEvent:
    trigger_step: int  # Simulation steps since scenario activation
    symbol: str
    price_change_pct: float
    news_item: Optional[str] = None

@dataclass
class MarketScenario:
    name: str
    description: str
    global_drift: float = 0.0
    global_volatility_multiplier: float = 1.0
    sector_multipliers: Dict[str, float] = field(default_factory=dict) # Symbol -> Drift
    news_templates: List[str] = field(default_factory=list)
    scheduled_events: List[ScenarioEvent] = field(default_factory=list)

# Preset Scenarios
SCENARIOS = {
    "NORMAL": MarketScenario(
        name="Normal Market",
        description="Standard market conditions with low drift and average volatility.",
        global_drift=0.0001,
        global_volatility_multiplier=1.0,
        news_templates=[
            "Market remains range-bound as investors digest earnings.",
            "Trading volume is average today.",
            "Analysts see steady growth ahead."
        ]
    ),
    "BULL_RALLY": MarketScenario(
        name="Bull Rally",
        description="Strong upward momentum across all sectors.",
        global_drift=0.0020, # Strong positive drift
        global_volatility_multiplier=1.2,
        news_templates=[
            "Stocks hit new all-time highs!",
            "Investors cheer strong economic data.",
            "Rally continues as tech leads the way."
        ]
    ),
    "BEAR_CRASH": MarketScenario(
        name="Bear Crash",
        description="Severe market correction.",
        global_drift=-0.0050, # Strong negative drift
        global_volatility_multiplier=3.0,
        news_templates=[
            "Market plunges amid recession fears.",
            "Panic selling hits Wall Street.",
            "Investors flee to safety as volatility spikes."
        ]
    ),
    "TECH_BOOM": MarketScenario(
        name="Tech Boom",
        description="Technology sector outperforms while others lag.",
        global_drift=0.0005,
        global_volatility_multiplier=1.5,
        sector_multipliers={
            "AAPL": 0.0030, "MSFT": 0.0030, "NVDA": 0.0040, "AMD": 0.0035,
            "NQ=F": 0.0025, "AWAV": 0.0050, "PLTR": 0.0040
        },
        news_templates=[
            "AI hype drives tech stocks higher.",
            "Semiconductor index surges on new chip demand.",
            "Tech earnings crush expectations."
        ]
    ),
    "MIDNIGHT_HAMMER": MarketScenario(
        name="Operation Midnight Hammer",
        description="Geopolitical energy shock. Oil spikes, stocks wobble.",
        global_drift=-0.0010,
        global_volatility_multiplier=2.5,
        sector_multipliers={
            "CL=F": 0.0100, # Oil spikes massively
            "XOM": 0.0050,  # Energy stocks up
            "GC=F": 0.0030, # Gold up (Safe haven)
            "LMT": 0.0040   # Defense up
        },
        news_templates=[
            "Oil prices surge amid geopolitical tensions.",
            "Energy sector rallies while broader market slips.",
            "Gold shines as safe-haven demand increases."
        ]
    ),
    "FRACTURED_OUROBOROS": MarketScenario(
        name="Fractured Ouroboros",
        description="Credit cycle collapse. High Yield spreads widen, Banks suffer.",
        global_drift=-0.0020,
        global_volatility_multiplier=2.0,
        sector_multipliers={
            "JPM": -0.0050, "BK": -0.0040, "KRE": -0.0080, # Banks hit hard
            "HYG": -0.0060, # High Yield down
            "GC=F": 0.0020  # Gold up
        },
        news_templates=[
            "Credit spreads widen to 52-week highs.",
            "Bank stocks under pressure as default risks rise.",
            "Liquidity concerns resurface in repo markets."
        ]
    )
}

def get_scenario(name: str) -> MarketScenario:
    # Lazy load external scenarios if requested one is missing
    key = name.upper().replace(" ", "_")
    if key not in SCENARIOS:
        load_external_scenarios()

    return SCENARIOS.get(key, SCENARIOS["NORMAL"])

def load_external_scenarios():
    """
    Scans the data/scenarios directory and updates the global SCENARIOS registry.
    """
    try:
        from core.market_data.scenario_loader import loader
        external = loader.load_all()
        SCENARIOS.update(external)
    except ImportError:
        pass # Loader might not be ready or circular dep
    except Exception as e:
        print(f"Error loading external scenarios: {e}")
