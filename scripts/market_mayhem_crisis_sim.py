import random
import math

class CrisisSimulator:
    def __init__(self, initial_portfolio_value=1000000):
        self.initial_value = initial_portfolio_value

    def _generate_path(self, days, volatility, drift, shock_day=None, shock_magnitude=0.0):
        """Generates a random walk price path with an optional shock."""
        values = [self.initial_value]
        current_val = self.initial_value

        for day in range(1, days + 1):
            daily_vol = random.gauss(drift, volatility)

            # Apply shock on specific day
            if shock_day and day == shock_day:
                daily_vol += shock_magnitude

            # Recovery phase after shock (simple mean reversion logic)
            if shock_day and day > shock_day and day < shock_day + 20:
                 if shock_magnitude < 0: # If it was a crash
                     daily_vol += abs(shock_magnitude) * 0.1 # Bounce back

            change_percent = daily_vol
            current_val = current_val * (1 + change_percent)
            values.append(int(current_val))

        return values

    def simulate_2008(self):
        """Simulates a Lehman-style systematic collapse."""
        # 60 days, slow bleed then massive drop
        return {
            "scenario": "2008 Financial Crisis",
            "description": "Systemic banking failure leading to credit freeze.",
            "dates": [f"Day {i}" for i in range(61)],
            "portfolio_values": self._generate_path(60, 0.01, -0.002, shock_day=45, shock_magnitude=-0.25),
            "benchmark_values": self._generate_path(60, 0.015, -0.005, shock_day=45, shock_magnitude=-0.35) # Benchmark falls harder
        }

    def simulate_1987(self):
        """Simulates Black Monday flash crash."""
        # 30 days, stable then one day -20%
        return {
            "scenario": "1987 Black Monday",
            "description": "Program trading cascade triggering massive liquidity void.",
            "dates": [f"Day {i}" for i in range(31)],
            "portfolio_values": self._generate_path(30, 0.008, 0.001, shock_day=20, shock_magnitude=-0.22),
            "benchmark_values": self._generate_path(30, 0.01, 0.001, shock_day=20, shock_magnitude=-0.25)
        }

    def simulate_covid(self):
        """Simulates 2020 Pandemic shutdown."""
        # 40 days, sharp drop, sharp recovery
        return {
            "scenario": "2020 Pandemic Lockdowns",
            "description": "Global economic shutdown followed by fiscal stimulus.",
            "dates": [f"Day {i}" for i in range(41)],
            "portfolio_values": self._generate_path(40, 0.02, -0.005, shock_day=15, shock_magnitude=-0.30),
            "benchmark_values": self._generate_path(40, 0.025, -0.008, shock_day=15, shock_magnitude=-0.35)
        }

    def simulate_generic_stress(self):
        """Simulates a standard recessionary bear market."""
        return {
            "scenario": "Standard Recession",
            "description": "Cyclical downturn with contracting earnings multiples.",
            "dates": [f"Day {i}" for i in range(91)],
            "portfolio_values": self._generate_path(90, 0.012, -0.001), # Slow drift down
            "benchmark_values": self._generate_path(90, 0.01, -0.002)
        }

    def get_simulation(self, report_date_str):
        """Factory method to pick the right simulation based on date."""
        year = report_date_str.split("-")[0]

        if year == "2008":
            return self.simulate_2008()
        elif year == "1987":
            return self.simulate_1987()
        elif year == "2020":
            return self.simulate_covid()
        else:
            # Default to None or Generic for others?
            # Strategy: Only return specific historical sims for matching years,
            # otherwise maybe a generic 'Stress Test' for future outlooks.
            if int(year) > 2024:
                return self.simulate_generic_stress()
            return None
