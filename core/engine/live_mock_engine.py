import json
import random
import time
import threading
import os
from copy import deepcopy

class LiveMockEngine:
    """
    A simulation engine that loads seed data and generates infinite,
    evolving market signals to mimic a live runtime environment.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LiveMockEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.data_path = os.path.join(os.path.dirname(__file__), 'live_seed_data.json')
        self.state = self._load_seed_data()
        self.last_update = time.time()
        self._initialized = True

    def _load_seed_data(self):
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            # Fallback if file not found
            return {
                "market_data": {"indices": {"SPX": {"price": 5000, "change_percent": 0.0}}},
                "headlines": [],
                "agent_thoughts": ["System initialized."]
            }

    def _drift_value(self, value, volatility=0.001):
        """Apply a random walk drift to a value."""
        change = value * volatility * (random.random() - 0.5)
        return value + change

    def get_market_pulse(self):
        """
        Returns the current state of the market with slight random mutations
        to simulate live ticker updates.
        """
        now = time.time()
        # Update state every call (simulation step)
        indices = self.state['market_data']['indices']

        updated_indices = {}
        for symbol, data in indices.items():
            new_price = self._drift_value(data['price'], volatility=0.0005)
            # Update the stored state so it evolves continuously
            indices[symbol]['price'] = new_price

            # Recalculate change percent roughly
            # (In a real sim, we'd track open price, but here we just drift the % slightly too)
            new_change = data['change_percent'] + (random.random() - 0.5) * 0.05
            indices[symbol]['change_percent'] = new_change

            updated_indices[symbol] = {
                "price": round(new_price, 2),
                "change_percent": round(new_change, 2),
                "volatility": data.get("volatility", 0)
            }

        return {
            "indices": updated_indices,
            "sectors": self.state['market_data']['sectors'],
            "timestamp": now
        }

    def get_agent_stream(self, limit=5):
        """
        Returns a stream of agent thoughts, occasionally generating a new one
        by mixing templates.
        """
        thoughts = self.state['agent_thoughts']
        headlines = self.state['headlines']

        # 20% chance to generate a new dynamic thought based on headlines
        if random.random() < 0.2 and headlines:
            topic = random.choice(headlines)['title']
            action = random.choice(["Analyzing impact of", "Correlating", "Hedging against", "Ignoring noise from"])
            new_thought = f"{action} '{topic}'..."
            thoughts.insert(0, new_thought)
            if len(thoughts) > 20: # Keep buffer size managed
                thoughts.pop()

        return thoughts[:limit]

    def get_synthesizer_score(self):
        """
        Calculates a 'System Confidence Score' based on market volatility and trends.
        """
        indices = self.state['market_data']['indices']
        spx_change = indices['SPX']['change_percent']
        vix = indices['VIX']['price']

        # Simple heuristic: High SPX & Low VIX = High Confidence
        base_score = 75
        score = base_score + (spx_change * 10) - ((vix - 15) * 2)

        # Clamp 0-100
        return max(0, min(100, round(score, 1)))

# Global singleton access
live_engine = LiveMockEngine()
