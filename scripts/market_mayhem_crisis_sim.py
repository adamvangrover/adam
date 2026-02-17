import hashlib
import random
import json

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

        random.seed(self.seed)

        days = list(range(1, scenario["duration"] + 1))
        market_values = [100.0]
        portfolio_values = [100.0]

        # Risk Multiplier: High Risk = higher beta
        beta = 1.0
        if self.risk_profile == "AGGRESSIVE": beta = 1.5
        elif self.risk_profile == "CONSERVATIVE": beta = 0.6

        current_market = 100.0
        current_portfolio = 100.0

        for day in days[1:]:
            # Market Movement
            daily_change = random.gauss(scenario["market_trend"], scenario["market_volatility"])

            # Apply Shock
            if day == scenario["shock_event_day"]:
                daily_change += scenario["shock_magnitude"]

            # Apply Recovery Bounce (simplified)
            if day > scenario["shock_event_day"] + 5:
                daily_change += 0.005 # Slight recovery bias

            current_market *= (1 + daily_change)

            # Portfolio Movement (Beta + Alpha/Idiosyncratic)
            portfolio_change = daily_change * beta + random.gauss(0, 0.02) # Alpha noise
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
