import json
import os
import re
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from core.market_data.scenarios import SCENARIOS, get_scenario, MarketScenario

class MarketDataManager:
    """
    Manages the 'Real-Time' market state for the Adam system.

    Architecture:
    - Source of Truth: showcase/js/market_snapshot.js (for frontend compatibility)
    - Fallback Protocol: Real Input -> Verified Historic -> Simulation -> Mock
    """

    SNAPSHOT_FILE = "showcase/js/market_snapshot.js"

    def __init__(self, filepath: str = None):
        self.filepath = filepath or self.SNAPSHOT_FILE
        self.state = self._load()
        self.history_cache = {} # Cache for simulation drift
        self.scenario_step_count = 0 # Track progress in current scenario

        # Restore scenario from state if available
        scenario_name = self.state.get("metadata", {}).get("active_scenario", "NORMAL")
        # Handle "Normal Market" vs "NORMAL" key mismatch (Scenario.name vs Scenario Key)
        # We search by name or key
        self.active_scenario = self._resolve_scenario(scenario_name)

    def _resolve_scenario(self, name_or_key: str) -> MarketScenario:
        # 1. Try Direct Key Match
        if name_or_key.upper() in SCENARIOS:
            return SCENARIOS[name_or_key.upper()]

        # 2. Try Name Match
        for key, scen in SCENARIOS.items():
            if scen.name == name_or_key:
                return scen

        return SCENARIOS["NORMAL"]

    def _load(self) -> Dict[str, Any]:
        """Loads the JSON state from the JS file wrapper."""
        if not os.path.exists(self.filepath):
            return self._create_default_state()

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Strip JS assignment: window.MARKET_SNAPSHOT = { ... };
                json_str = re.sub(r'^window\.MARKET_SNAPSHOT\s*=\s*', '', content)
                json_str = re.sub(r';\s*$', '', json_str)
                return json.loads(json_str)
        except Exception as e:
            print(f"Error loading market state: {e}. Reverting to default.")
            return self._create_default_state()

    def _save(self):
        """Saves the JSON state back to the JS file wrapper."""
        # Update metadata with scenario info
        if "metadata" not in self.state:
            self.state["metadata"] = {}

        self.state["metadata"]["active_scenario"] = self.active_scenario.name
        self.state["metadata"]["generated_at"] = datetime.now().isoformat()

        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"window.MARKET_SNAPSHOT = {json.dumps(self.state, indent=2)};")

    def _create_default_state(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "description": "Default Initial State",
                "active_scenario": "Normal Market"
            },
            "market_data": {},
            "news_feed": []
        }

    def get_price(self, symbol: str) -> Optional[float]:
        return self.state.get("market_data", {}).get(symbol, {}).get("price")

    def set_scenario(self, scenario_name: str) -> str:
        """Sets the active market scenario."""
        scenario = get_scenario(scenario_name)
        self.active_scenario = scenario
        self.scenario_step_count = 0 # Reset counter

        # Inject a news item announcing the scenario (if not Normal)
        if scenario_name.upper() != "NORMAL":
            headline = f"MARKET ALERT: {scenario.description}"
            self.add_news(headline, "NEUTRAL", source="System Scenarios")

        self._save()
        return f"Active Scenario set to: {scenario.name}"

    def update_symbol(self, symbol: str, price: float, volume: int = None, source: str = "manual"):
        """
        Updates a single symbol with 'Real' data.
        Updates history and calculates change %.
        """
        market_data = self.state.setdefault("market_data", {})

        current_data = market_data.get(symbol, {
            "price": price,
            "volume": volume or 1000,
            "history": [],
            "change_pct": 0.0
        })

        # History Management
        history = current_data.get("history", [])
        prev_price = current_data["price"]

        # If this is a new update (price changed), append to history
        if prev_price != price:
            history.append(prev_price)
            if len(history) > 50: # Keep sparkline manageable
                history.pop(0)

        # Calculate Change % based on the start of the history window
        start_price = history[0] if history else price
        if start_price == 0: start_price = price # Avoid div by zero
        change_pct = round(((price - start_price) / start_price) * 100, 2)

        # Update State
        market_data[symbol] = {
            "price": price,
            "volume": volume or current_data.get("volume", 1000),
            "change_pct": change_pct,
            "history": history,
            "last_updated": datetime.now().isoformat(),
            "source": source
        }

        self._save()
        return market_data[symbol]

    def ingest_bulk_update(self, data_dict: Dict[str, float], source: str = "bulk"):
        """Ingests a dictionary of {Symbol: Price}."""
        for sym, price in data_dict.items():
            self.update_symbol(sym, price, source=source)

    def simulate_step(self, symbols: List[str] = None):
        """
        Applies a simulation step based on the ACTIVE SCENARIO.
        Executes drift, volatility, and scheduled events.
        """
        self.scenario_step_count += 1
        target_symbols = symbols or list(self.state.get("market_data", {}).keys())

        # 1. Check for Scheduled Events
        triggered_events = [e for e in self.active_scenario.scheduled_events if e.trigger_step == self.scenario_step_count]

        for event in triggered_events:
            if event.symbol in target_symbols or event.symbol == "ALL":
                # Apply Shock
                targets = target_symbols if event.symbol == "ALL" else [event.symbol]
                for sym in targets:
                    data = self.state["market_data"].get(sym)
                    if not data: continue
                    new_price = data["price"] * (1 + event.price_change_pct)
                    self.update_symbol(sym, round(new_price, 2), source="Event")

                # Post News
                if event.news_item:
                    sentiment = "POSITIVE" if event.price_change_pct > 0 else "NEGATIVE"
                    self.add_news(event.news_item, sentiment, source="Market Event")

        # 2. Standard Simulation Loop
        for symbol in target_symbols:
            data = self.state["market_data"].get(symbol)
            if not data: continue

            current_price = data["price"]

            # Scenario Parameters
            drift = self.active_scenario.global_drift
            volatility_mult = self.active_scenario.global_volatility_multiplier

            # Sector Overrides
            # Check if symbol matches specific overrides (exact match or substring like 'BTC')
            for key, val in self.active_scenario.sector_multipliers.items():
                if key == symbol or (key in symbol):
                    drift = val
                    break

            base_volatility = 0.0005
            volatility = base_volatility * volatility_mult

            shock = random.normalvariate(0, 1)

            new_price = current_price * (1 + drift + volatility * shock)

            # Update state (marking source as 'simulation')
            self.update_symbol(symbol, round(new_price, 2), source=f"Sim:{self.active_scenario.name}")

        # Occasionally inject scenario news
        if self.active_scenario.news_templates and random.random() < 0.05:
            template = random.choice(self.active_scenario.news_templates)
            self.add_news(template, "NEUTRAL", source="Market Wire")

    def add_news(self, headline: str, sentiment: str, source: str = "System"):
        """Injects a news item."""
        news_feed = self.state.setdefault("news_feed", [])

        item = {
            "headline": headline,
            "sentiment": sentiment,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }

        # Dedup
        if not any(n["headline"] == headline for n in news_feed):
            news_feed.insert(0, item)
            if len(news_feed) > 20:
                news_feed.pop()

        self._save()

# Example Usage
if __name__ == "__main__":
    mgr = MarketDataManager()
    print(f"Current Scenario: {mgr.active_scenario.name}")
    mgr.set_scenario("BULL_RALLY")
    mgr.simulate_step()
    print("Simulated step under BULL_RALLY.")
