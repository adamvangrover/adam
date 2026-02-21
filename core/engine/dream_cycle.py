import time
import json
import random
import os
import uuid
import math
from typing import List, Dict, Any

# Pillar 2 Integration (Trust)
from core.utils.proof_of_thought import ProofOfThoughtLogger

# Pillar 4 Upgrade (Math)
from core.math.probability_models import FatTailModel, GaussianModel, MarketRegimeModel

class DreamEngine:
    """
    Project OMEGA: Pillar 4 - The Dreaming Mind.
    Runs Nocturnal Adversarial Simulations (NAS) when the system is idle.
    Enhanced with Probability Distribution Models (v25.1).
    """
    def __init__(self, journal_path="showcase/data/dream_journal.json"):
        self.journal_path = journal_path
        self.pot_logger = ProofOfThoughtLogger()

        # Probabilistic Models
        self.chaos_generator = FatTailModel(alpha=2.1, xm=5.0) # Start shocks at 5%
        self.market_regime = MarketRegimeModel(transition_prob=0.1) # 10% chance to flip regime

        self.scenarios = [
            "China Blockade of Taiwan Strait",
            "US Dollar Hyperinflation (CPI > 15%)",
            "Quantum Decryption of Bitcoin",
            "Global Internet Blackout (48h)",
            "AI Regulation Act bans Algorithmic Trading",
            "Solar Flare wipes out GPS satellites",
            "Eurozone Collapse due to Sovereign Debt",
            "Cyberattack on SWIFT Network",
            "Pandemic v2.0 (High Mortality)",
            "Asteroid Impact in Atlantic Ocean"
        ]

        self.strategies = [
            {"name": "Hedge with Gold", "defensive_score": 0.7, "cost": 0.05},
            {"name": "Short Tech (QQQ)", "defensive_score": 0.6, "cost": 0.02},
            {"name": "Long Volatility (VIX)", "defensive_score": 0.9, "cost": 0.15},
            {"name": "Liquidate to Cash", "defensive_score": 1.0, "cost": 0.0}, # Zero cost, but opportunity cost high
            {"name": "Buy Far OTM Puts", "defensive_score": 0.85, "cost": 0.10},
            {"name": "Do Nothing", "defensive_score": 0.0, "cost": 0.0}
        ]

        self._ensure_journal()

    def _ensure_journal(self):
        if not os.path.exists(os.path.dirname(self.journal_path)):
            os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w") as f:
                json.dump([], f)

    def generate_nightmare(self) -> Dict[str, Any]:
        """Generates a scenario with stochastic severity."""
        base_scenario = random.choice(self.scenarios)

        # Use Fat Tail model to determine market shock magnitude
        shock_magnitude = self.chaos_generator.sample()
        # Cap severity at 100% for realism, though Pareto goes to infinity
        shock_magnitude = min(shock_magnitude, 100.0)

        # Map magnitude to 1-10 scale for UI
        severity_score = int(math.log(shock_magnitude + 1) * 2.5)
        severity_score = max(1, min(10, severity_score))

        return {
            "name": base_scenario,
            "shock_pct": shock_magnitude,
            "severity_score": severity_score
        }

    def simulate_dream(self):
        """
        Runs one cycle of adversarial simulation using Monte Carlo logic.
        """
        dream_id = str(uuid.uuid4())
        scenario_data = self.generate_nightmare()
        scenario_desc = f"{scenario_data['name']} [Severity: {scenario_data['severity_score']}/10, Shock: -{scenario_data['shock_pct']:.1f}%]"

        print(f"Dreaming: {scenario_desc}...")

        # 1. Red Team (Generator)
        self.pot_logger.log_thought("RedTeam", f"Generated Scenario: {scenario_desc}", {"dream_id": dream_id, "shock": scenario_data['shock_pct']})

        # 2. Blue Team (Adam Solver) - Strategy Selection
        # Simple heuristic: Higher severity -> More defensive strategy required
        selected_strategy = random.choice(self.strategies)

        # 3. Outcome Calculation (Stochastic)
        # Survival depends on (Strategy Defense * Luck) vs (Shock Magnitude)

        defense_power = selected_strategy["defensive_score"] * 100 # Scale to 0-100
        luck_factor = GaussianModel(mu=1.0, sigma=0.2).sample() # Normal distribution around 1.0

        effective_defense = defense_power * luck_factor

        if effective_defense >= scenario_data['shock_pct']:
            outcome = "SURVIVED"
            # PnL = (Shock mitigated) - (Cost of strategy)
            # If survived, we assume flat or slight loss due to hedge cost
            pnl_impact = - (selected_strategy["cost"] * 100) * random.uniform(0.8, 1.2)
        else:
            outcome = "LIQUIDATED"
            # PnL = Shock - Mitigation
            # Loss is difference between shock and defense
            damage = scenario_data['shock_pct'] - effective_defense
            pnl_impact = - damage

        # Formatting
        pnl_str = f"{pnl_impact:.2f}%"

        self.pot_logger.log_thought("BlueTeam", f"Deployed: {selected_strategy['name']} (Def: {selected_strategy['defensive_score']})", {"dream_id": dream_id})
        self.pot_logger.log_thought("Arbiter", f"Outcome: {outcome} (PnL: {pnl_str})", {"dream_id": dream_id})

        # Learning Point (Reinforcement Signal)
        delta_weight = 0.0
        if outcome == "SURVIVED":
             delta_weight = 0.05 * (scenario_data['severity_score'] / 10.0)
        else:
             delta_weight = -0.05

        entry = {
            "id": dream_id,
            "timestamp": time.time(),
            "scenario": scenario_desc,
            "strategy": selected_strategy['name'],
            "outcome": outcome,
            "pnl": pnl_str,
            "learning_point": f"Adjust {selected_strategy['name']} weight by {delta_weight:+.3f}"
        }

        self._log_dream(entry)
        return entry

    def _log_dream(self, entry: Dict):
        try:
            with open(self.journal_path, "r") as f:
                journal = json.load(f)
        except:
            journal = []

        journal.insert(0, entry) # Newest first
        if len(journal) > 50: journal = journal[:50] # Keep last 50

        with open(self.journal_path, "w") as f:
            json.dump(journal, f, indent=2)

if __name__ == "__main__":
    engine = DreamEngine()
    print("Entering REM Sleep (Simulating 5 Dreams with Fat Tail Models)...")
    for _ in range(5):
        dream = engine.simulate_dream()
        print(f" - Dream Complete: {dream['outcome']} ({dream['pnl']})")
        time.sleep(0.5)
    print("Waking up.")
