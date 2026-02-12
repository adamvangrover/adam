import json
import random
import time
import threading
import os
import uuid
from copy import deepcopy
from core.engine.consensus_engine import ConsensusEngine
from core.utils.narrative_weaver import NarrativeWeaver

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
        self.consensus = ConsensusEngine()
        self.weaver = NarrativeWeaver()
        self._initialized = True

    def _load_seed_data(self):
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)

                # Bolt Optimization: Convert legacy string thoughts to objects with IDs
                # This enables stable key rendering on the frontend
                thoughts = data.get('agent_thoughts', [])
                if thoughts and isinstance(thoughts[0], str):
                    data['agent_thoughts'] = [
                        {"id": str(uuid.uuid4()), "text": t} for t in thoughts
                    ]
                return data
        except Exception as e:
            # Fallback if file not found
            return {
                "market_data": {"indices": {"SPX": {"price": 5000, "change_percent": 0.0}}},
                "headlines": [],
                "agent_thoughts": [{"id": str(uuid.uuid4()), "text": "System initialized."}]
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
            new_thought_text = f"{action} '{topic}'..."

            # Bolt Optimization: Generate unique ID for new thoughts
            new_thought = {
                "id": str(uuid.uuid4()),
                "text": new_thought_text
            }

            thoughts.insert(0, new_thought)
            if len(thoughts) > 20: # Keep buffer size managed
                thoughts.pop()

        return thoughts[:limit]

    def get_synthesizer_score(self):
        """
        Calculates a 'System Confidence Score' using the ConsensusEngine.
        It generates synthetic agent signals based on market data.
        """
        indices = self.state['market_data']['indices']
        spx_change = indices['SPX']['change_percent']
        vix = indices['VIX']['price']

        # Generate Synthetic Agent Signals
        signals = []

        # 1. Risk Officer (Conservative)
        risk_vote = "REJECT" if vix > 18 else "APPROVE"
        signals.append({
            "agent": "RiskOfficer",
            "vote": risk_vote,
            "confidence": 0.9,
            "weight": 2.0,
            "reason": f"VIX is at {vix:.2f}"
        })

        # 2. Trend Follower (Aggressive)
        trend_vote = "APPROVE" if spx_change > 0 else "REJECT"
        signals.append({
            "agent": "TrendFollower",
            "vote": trend_vote,
            "confidence": 0.7 + (abs(spx_change) * 0.1),
            "weight": 1.5,
            "reason": f"SPX change is {spx_change:.2f}%"
        })

        # 3. Macro Sentinel (Random/Cyclical)
        # Flip a coin based on time (simulating external news flow)
        macro_bullish = int(time.time()) % 60 < 30
        signals.append({
            "agent": "MacroSentinel",
            "vote": "APPROVE" if macro_bullish else "REJECT",
            "confidence": 0.6,
            "weight": 1.0,
            "reason": "Global macro cycle interpretation"
        })

        # 4. Blindspot Agent (Contrarian Check)
        # Protocol: ADAM-V-NEXT - Enterprise Integration
        # Only weighs in if volatility is extremely low (Coiled Spring)
        if indices['VIX']['price'] < 12.0:
             signals.append({
                "agent": "BlindspotScanner",
                "vote": "REJECT", # Expect volatility spike
                "confidence": 0.85,
                "weight": 2.5, # High weight for ignored risks
                "reason": "Volatility compression detected (VIX < 12)"
            })

        # Run Consensus
        result = self.consensus.evaluate(signals)

        # Map Consensus Score (-1 to 1) to 0-100 scale
        # -1 -> 0, 0 -> 50, 1 -> 100
        normalized_score = ((result['score'] + 1) / 2) * 100
        final_score = max(0, min(100, round(normalized_score, 1)))

        # Protocol: ADAM-V-NEXT - Narrative Intelligence
        # Use Weaver to generate the Mission Brief
        sentiment_key = "NEUTRAL"
        if final_score > 60: sentiment_key = "BULLISH"
        elif final_score < 40: sentiment_key = "BEARISH"

        narrative_ctx = {
            "sentiment": sentiment_key,
            "driver": "Macro Cycle" if macro_bullish else "VIX Volatility",
            "risk_factor": "Liquidity Compression" if vix < 12 else "Market Noise",
            "sector": "Broad Market"
        }

        # Enrich the rationale with the woven story
        result['rationale'] = self.weaver.weave(narrative_ctx)
        result['normalized_score'] = final_score

        return result

    def get_geopolitical_event(self):
        """
        Generates a geopolitical event string and its stability impact.
        Format: "[SovereignName][INTERNAL_MEMO]: EVENT_TYPE: Details..."
        """
        # Lazy load sovereigns
        if not hasattr(self, 'nexus_sovereigns'):
            try:
                # Try to load from generated simulation data
                path = os.path.join(os.path.dirname(__file__), '../../showcase/data/nexus_simulation.json')
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        self.nexus_sovereigns = [n['label'] for n in data['graph']['nodes']]
                else:
                    self.nexus_sovereigns = ["NorthAm-Market-1", "EuroZone-Digital-2", "AsiaPac-Resource-3"]
            except Exception as e:
                # Fallback
                self.nexus_sovereigns = ["NorthAm-Market-1", "EuroZone-Digital-2", "AsiaPac-Resource-3"]

        sov = random.choice(self.nexus_sovereigns)

        event_types = [
            ("Cyber Operation", -0.05, "Targeting critical infrastructure"),
            ("Military Maneuver", -0.1, "Mobilizing assets near contested zone"),
            ("Trade Sanction", -0.02, "Imposing tariffs on key exports"),
            ("Diplomatic Summit", 0.05, "Hosting peace talks"),
            ("Energy Crisis", -0.08, "Experiencing supply chain disruptions"),
            ("Strategic Alliance", 0.03, "Signing mutual defense treaty"),
            ("Civil Unrest", -0.07, "Protests erupting in capital"),
            ("Currency Devaluation", -0.04, "Central bank intervenes to stabilize currency"),
            ("Ransomware Attack", -0.06, "Paralyzing national banking systems"),
            ("Data Leak", -0.03, "Exposing sensitive government communications"),
            ("Flash Crash", -0.09, "Algo-trading glitch causes market plummet"),
            ("Liquidity Crisis", -0.07, "Interbank lending freezes unexpectedly"),
            ("Satellite outage", -0.04, "Global GPS network disruption reported"),
            ("AI Hallucination", -0.02, "Automated defense systems trigger false alarm")
        ]

        etype, impact, detail_base = random.choice(event_types)

        # Add slight variation to details
        if random.random() < 0.5:
            detail = detail_base + "."
        else:
            target = random.choice(self.nexus_sovereigns)
            detail = f"{detail_base} involving {target}."

        text = f"[{sov}][INTERNAL_MEMO]: {etype.upper()}: {detail}"

        return {
            "text": text,
            "stability_delta": impact
        }

# Global singleton access
live_engine = LiveMockEngine()
