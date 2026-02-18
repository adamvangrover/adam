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
            "2000_DOTCOM_BUBBLE": {
                "duration": 45,
                "market_volatility": 0.04,
                "market_trend": -0.015,
                "shock_event_day": 10,
                "shock_magnitude": -0.10
            },
            "2020_COVID": {
                "duration": 40,
                "market_volatility": 0.06,
                "market_trend": -0.015,
                "shock_event_day": 20,
                "shock_magnitude": -0.12
            },
            "2022_INFLATION_SHOCK": {
                "duration": 60,
                "market_volatility": 0.03,
                "market_trend": -0.01,
                "shock_event_day": 30,
                "shock_magnitude": -0.05
            }
        }

    def calculate_metrics(self, values):
        """Calculates Max Drawdown, Volatility, and Total Return."""
        if not values:
            return {
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "total_return": 0.0
            }

        # Max Drawdown
        peak = values[0]
        max_dd = 0.0
        for val in values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Volatility (Annualized approximation)
        returns = []
        for i in range(1, len(values)):
            r = (values[i] - values[i-1]) / values[i-1]
            returns.append(r)

        if returns:
            mean_r = sum(returns) / len(returns)
            variance = sum([(x - mean_r)**2 for x in returns]) / len(returns)
            volatility = (variance**0.5) * (252**0.5)
        else:
            volatility = 0.0

        return {
            "max_drawdown": round(max_dd * 100, 2),
            "volatility": round(volatility * 100, 2),
            "total_return": round((values[-1] - values[0]) / values[0] * 100, 2)
        }

    def run_simulation(self, scenario_name: str):
        """
        Runs a simulation for a given crisis scenario.
        Returns a dict with labels (Days), two series (Portfolio, Market), and metrics.
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

        market_metrics = self.calculate_metrics(market_values)
        portfolio_metrics = self.calculate_metrics(portfolio_values)

        return {
            "scenario": scenario_name,
            "labels": days,
            "market": market_values,
            "portfolio": portfolio_values,
            "metrics": {
                "market": market_metrics,
                "portfolio": portfolio_metrics
            }
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
