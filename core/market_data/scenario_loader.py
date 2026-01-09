import os
import yaml
import json
import logging
from typing import Dict, List, Optional
from glob import glob
from core.market_data.scenarios import MarketScenario, ScenarioEvent

logger = logging.getLogger(__name__)

class ScenarioLoader:
    """
    Responsible for loading MarketScenario definitions from external files (YAML/JSON).
    Follows the 'Additive' philosophy by extending the hardcoded scenario base.
    """

    def __init__(self, scenarios_dir: str = "data/scenarios"):
        self.scenarios_dir = scenarios_dir

    def load_all(self) -> Dict[str, MarketScenario]:
        """
        Scans the configured directory and loads all valid scenario definitions.
        Returns a dictionary {SCENARIO_NAME: MarketScenario}.
        """
        loaded_scenarios = {}

        if not os.path.exists(self.scenarios_dir):
            logger.warning(f"Scenario directory {self.scenarios_dir} not found. Skipping file load.")
            return loaded_scenarios

        # Load YAML
        yaml_files = glob(os.path.join(self.scenarios_dir, "*.yaml")) + glob(os.path.join(self.scenarios_dir, "*.yml"))
        for fpath in yaml_files:
            try:
                scenario = self._load_yaml(fpath)
                if scenario:
                    loaded_scenarios[scenario.name.upper().replace(" ", "_")] = scenario
            except Exception as e:
                logger.error(f"Failed to load scenario from {fpath}: {e}")

        # Load JSON
        json_files = glob(os.path.join(self.scenarios_dir, "*.json"))
        for fpath in json_files:
            try:
                scenario = self._load_json(fpath)
                if scenario:
                    loaded_scenarios[scenario.name.upper().replace(" ", "_")] = scenario
            except Exception as e:
                logger.error(f"Failed to load scenario from {fpath}: {e}")

        logger.info(f"Loaded {len(loaded_scenarios)} external scenarios.")
        return loaded_scenarios

    def _load_yaml(self, filepath: str) -> Optional[MarketScenario]:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            return self._parse_dict(data)

    def _load_json(self, filepath: str) -> Optional[MarketScenario]:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return self._parse_dict(data)

    def _parse_dict(self, data: Dict) -> Optional[MarketScenario]:
        """
        Validates and converts a raw dictionary into a MarketScenario object.
        Applies defaults where necessary.
        """
        required_keys = ["name", "description"]
        if not all(k in data for k in required_keys):
            logger.warning(f"Scenario missing required keys ({required_keys}): {data}")
            return None

        # Parse Events
        events = []
        if "scheduled_events" in data:
            for evt in data["scheduled_events"]:
                try:
                    events.append(ScenarioEvent(
                        trigger_step=evt["step"],
                        symbol=evt["symbol"],
                        price_change_pct=evt["change"],
                        news_item=evt.get("news")
                    ))
                except KeyError as e:
                    logger.warning(f"Skipping malformed event in {data['name']}: missing {e}")

        # Extract fields with safe defaults
        return MarketScenario(
            name=data["name"],
            description=data["description"],
            global_drift=data.get("global_drift", 0.0),
            global_volatility_multiplier=data.get("global_volatility_multiplier", 1.0),
            sector_multipliers=data.get("sector_multipliers", {}),
            news_templates=data.get("news_templates", []),
            scheduled_events=events
        )

# Singleton helper
loader = ScenarioLoader()
