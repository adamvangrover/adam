import json
import os
import time
from typing import Dict, Any

class ScenarioEngine:
    """
    Simulates historical market conditions by replaying pre-defined scenario datasets.
    """
    def __init__(self):
        self.scenarios = {
            "2008_CRASH": self._load_2008_crash()
        }
        self.active_scenario = None
        self.step = 0

    def _load_2008_crash(self):
        """Hardcoded 2008 Crash Scenario Data (Lehman Weekend)."""
        return {
            "id": "2008_CRASH",
            "name": "Lehman Brothers Collapse (Sep 2008)",
            "description": "Global liquidity freeze and massive credit dislocation.",
            "timeline": [
                {
                    "timestamp_offset": 0,
                    "indices": {"SPX": {"price": 1251.70, "change_percent": -3.4}, "VIX": {"price": 25.66, "change_percent": 12.5}},
                    "headlines": [{"source": "WSJ", "title": "Lehman Brothers Files for Bankruptcy", "sentiment": "negative"}]
                },
                {
                    "timestamp_offset": 10,
                    "indices": {"SPX": {"price": 1192.70, "change_percent": -4.7}, "VIX": {"price": 31.70, "change_percent": 23.5}},
                    "headlines": [{"source": "Bloomberg", "title": "AIG Seeks Fed Rescue as Credit Markets Seize", "sentiment": "negative"}]
                },
                {
                    "timestamp_offset": 20,
                    "indices": {"SPX": {"price": 1130.00, "change_percent": -5.2}, "VIX": {"price": 36.5, "change_percent": 15.1}},
                    "headlines": [{"source": "CNBC", "title": "Money Market Funds Break the Buck", "sentiment": "negative"}]
                }
            ]
        }

    def set_scenario(self, scenario_key: str):
        if scenario_key in self.scenarios:
            self.active_scenario = self.scenarios[scenario_key]
            self.step = 0
            return True
        return False

    def get_pulse(self) -> Dict[str, Any]:
        """
        Returns the current step of the active scenario, looping if necessary.
        """
        if not self.active_scenario:
            return {}

        timeline = self.active_scenario['timeline']
        # Loop through timeline based on step
        current_data = timeline[self.step % len(timeline)]

        # Advance step
        self.step += 1

        return {
            "indices": current_data['indices'],
            "sectors": {"Financials": {"sentiment": -0.9, "trend": "bearish"}}, # Static for now
            "headlines": current_data['headlines'],
            "timestamp": time.time(),
            "scenario_mode": True,
            "scenario_name": self.active_scenario['name']
        }
