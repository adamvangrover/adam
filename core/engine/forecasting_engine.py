import numpy as np
import json
import os
from datetime import datetime, timedelta

class ForecastingEngine:
    """
    Provides forward-looking probabilistic projections using Monte Carlo simulations.
    Generates 'Confidence Bands' (Fan Charts) for risk visualization.
    """

    def __init__(self):
        pass

    def generate_forecast(self, symbol: str, history: list, days: int = 30, simulations: int = 1000):
        """
        Generates a 30-day forecast with 95% and 80% confidence intervals.

        Args:
            symbol: Ticker symbol
            history: List of historical prices (floats) or dicts with 'close'
            days: Number of days to forecast

        Returns:
            Dict containing dates, mean path, and confidence bands.
        """
        if not history:
            return {}

        # Extract prices
        prices = [h['close'] if isinstance(h, dict) else h for h in history]
        last_price = prices[-1]

        # Calculate recent volatility (simple std dev of log returns over last 30 days)
        if len(prices) > 30:
            recent = prices[-30:]
            returns = np.diff(np.log(recent))
            daily_vol = np.std(returns)
            # Add a "Risk Premium" drift
            daily_drift = np.mean(returns)
        else:
            daily_vol = 0.015 # Default fallback
            daily_drift = 0.0005

        # Monte Carlo Simulation (Bolt Optimized: Vectorized)
        # Generate all shocks at once: shape (simulations, days)
        # Reuse array for simulation paths to minimize allocations
        simulation_paths = np.random.normal(0, daily_vol, (simulations, days))

        # Cumulative sum of log returns (In-place optimization)
        simulation_paths += daily_drift
        np.cumsum(simulation_paths, axis=1, out=simulation_paths)

        # Calculate price paths (In-place)
        np.exp(simulation_paths, out=simulation_paths)
        simulation_paths *= last_price

        # Calculate Percentiles (Vectorized)
        mean_path = np.mean(simulation_paths, axis=0)
        upper_95 = np.percentile(simulation_paths, 97.5, axis=0)
        lower_95 = np.percentile(simulation_paths, 2.5, axis=0)
        upper_80 = np.percentile(simulation_paths, 90, axis=0)
        lower_80 = np.percentile(simulation_paths, 10, axis=0)

        # Generate Dates
        start_date = datetime.now()
        dates = [(start_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(days)]

        return {
            "symbol": symbol,
            "dates": dates,
            "mean": np.round(mean_path, 2).tolist(),
            "upper_95": np.round(upper_95, 2).tolist(),
            "lower_95": np.round(lower_95, 2).tolist(),
            "upper_80": np.round(upper_80, 2).tolist(),
            "lower_80": np.round(lower_80, 2).tolist()
        }

# Singleton for ease of import
forecasting_engine = ForecastingEngine()
