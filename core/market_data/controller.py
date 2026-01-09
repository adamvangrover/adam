from typing import Dict, Any
from core.market_data.manager import MarketDataManager

class ScenarioController:
    """
    Facade for controlling the Market Scenario Engine.
    Designed for integration into the MetaOrchestrator or other high-level agents.
    """

    def __init__(self):
        self.manager = MarketDataManager()

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "scenario": self.manager.active_scenario.name,
            "description": self.manager.active_scenario.description,
            "drift": self.manager.active_scenario.global_drift,
            "volatility": self.manager.active_scenario.global_volatility_multiplier
        }

    def activate_scenario(self, scenario_name: str) -> str:
        """
        Activates a specific market scenario (e.g. 'BEAR_CRASH').
        """
        return self.manager.set_scenario(scenario_name)

    def trigger_pulse(self):
        """
        Manually triggers a single simulation step.
        Useful for "Turn-based" simulation updates.
        """
        self.manager.simulate_step()
        return f"Pulse triggered for {len(self.manager.state.get('market_data', {}))} assets."

    def inject_news(self, headline: str, sentiment: str = "NEUTRAL", source: str = "Controller"):
        self.manager.add_news(headline, sentiment, source)
