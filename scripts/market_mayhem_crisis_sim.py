import hashlib
import random
import json
import math

class CrisisSimulator:
    def __init__(self, seed: str, risk_profile: str = "BALANCED"):
        """
        Initializes the Crisis Simulator with a deterministic seed.
        """
        self.seed = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        self.risk_profile = risk_profile.upper()
        self.scenarios = {
            "2008_LEHMAN": {
                "duration": 30,
                "market_volatility": 0.05,
                "market_trend": -0.02,
                "shock_event_day": 15,
                "shock_magnitude": -0.15
            },
            "1987_BLACK_MONDAY": {
                "duration": 20,
                "market_volatility": 0.08,
                "market_trend": -0.01,
                "shock_event_day": 10,
                "shock_magnitude": -0.22
            },
            "2020_COVID": {
                "duration": 40,
                "market_volatility": 0.06,
                "market_trend": -0.015,
                "shock_event_day": 20,
                "shock_magnitude": -0.12
            },
             "2000_DOTCOM": {
                "duration": 60,
                "market_volatility": 0.04,
                "market_trend": -0.01,
                "shock_event_day": 30,
                "shock_magnitude": -0.10
            },
            "2022_INFLATION": {
                "duration": 50,
                "market_volatility": 0.03,
                "market_trend": -0.005,
                "shock_event_day": 25,
                "shock_magnitude": -0.08
            },
            "2025_AI_SHOCK": {
                "duration": 35,
                "market_volatility": 0.07,
                "market_trend": -0.03,
                "shock_event_day": 18,
                "shock_magnitude": -0.18
            }
        }

    def run_simulation(self, scenario_name: str):
        """
        Runs a simulation for a given crisis scenario.
        Returns a dict with labels (Days) and two series (Portfolio, Market).
        """
        scenario = self.scenarios.get(scenario_name)
        if not scenario:
            return None

        # Seed per scenario to ensure different outcomes for different scenarios with same main seed
        scenario_seed = self.seed + sum(ord(c) for c in scenario_name)
        random.seed(scenario_seed)

        days = list(range(1, scenario["duration"] + 1))
        market_values = [100.0]
        portfolio_values = [100.0]

        # Risk Multiplier: High Risk = higher beta
        beta = 1.0
        if self.risk_profile == "AGGRESSIVE": beta = 1.5
        elif self.risk_profile == "CONSERVATIVE": beta = 0.6

        current_market = 100.0
        current_portfolio = 100.0

        base_vol = scenario["market_volatility"]
        shock_day = scenario["shock_event_day"]

        for day in days[1:]:
            # Volatility ramps up as we approach shock, spikes at shock, then slowly decays
            if day < shock_day:
                # Pre-shock jitter: Volatility increases linearly
                current_vol = base_vol * (1 + (day / shock_day) * 0.5)
            elif day == shock_day:
                 current_vol = base_vol * 3.0 # Spike
            else:
                 # Post-shock decay
                 days_since = day - shock_day
                 current_vol = base_vol * (1 + 2.0 * math.exp(-days_since / 5.0))

            # Market Movement
            daily_change = random.gauss(scenario["market_trend"], current_vol)

            # Apply Shock
            if day == shock_day:
                daily_change += scenario["shock_magnitude"]

            # Apply Recovery Bounce (simplified)
            if day > shock_day + 5:
                daily_change += 0.005 # Slight recovery bias

            current_market *= (1 + daily_change)

            # Portfolio Movement (Beta + Alpha/Idiosyncratic)
            # Alpha is random but slightly correlated to volatility (higher vol = more dispersion)
            alpha = random.gauss(0, 0.02 * (current_vol / base_vol))
            portfolio_change = daily_change * beta + alpha
            current_portfolio *= (1 + portfolio_change)

            market_values.append(round(current_market, 2))
            portfolio_values.append(round(current_portfolio, 2))

        return {
            "scenario": scenario_name,
            "labels": days,
            "market": market_values,
            "portfolio": portfolio_values
        }

    def run_all_scenarios(self):
        """Runs all defined crisis scenarios."""
        results = {}
        for name in self.scenarios:
            results[name] = self.run_simulation(name)
        return results

if __name__ == "__main__":
    # Test Run
    sim = CrisisSimulator("test_seed", "AGGRESSIVE")
    print(json.dumps(sim.run_all_scenarios(), indent=2))
