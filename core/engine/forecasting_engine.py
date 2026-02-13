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

    def generate_forecast(self, symbol: str, history: list, days: int = 30, simulations: int = 1000,
                         sentiment_score: float = 0.0, conviction_score: float = 0.0):
        """
        Generates a 30-day forecast with 95% and 80% confidence intervals.
        Integrates subjective views (Sentiment & Conviction) via drift adjustment.

        Args:
            symbol: Ticker symbol
            history: List of historical prices (floats) or dicts with 'close'
            days: Number of days to forecast
            sentiment_score: -1.0 (Bearish) to 1.0 (Bullish). Default 0.0 (Neutral).
            conviction_score: 0.0 (Low) to 1.0 (High). Default 0.0 (None).

        Returns:
            Dict containing dates, mean path, confidence bands, and adjustment metadata.
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
            # Add a "Risk Premium" drift based on historical mean
            base_drift = np.mean(returns)
        else:
            daily_vol = 0.015 # Default fallback
            base_drift = 0.0005

        # --- Conviction & Sentiment Adjustment (Black-Litterman Style View Integration) ---
        # Logic: If we have conviction, shift the distribution mean (Drift)
        # magnitude = 0.1 sigma per day * sentiment * conviction
        # Example: Bullish (1.0) High Conviction (1.0) -> +0.1 sigma drift per day
        view_drift_adjustment = (daily_vol * 0.1) * sentiment_score * conviction_score

        # Logic: High conviction slightly reduces uncertainty (Volatility Compression)
        # max_reduction = 20%
        vol_reduction_factor = 1.0 - (conviction_score * 0.2)
        adjusted_vol = daily_vol * vol_reduction_factor

        final_daily_drift = base_drift + view_drift_adjustment

        # Monte Carlo Simulation (Bolt Optimized: Vectorized)
        # Generate all shocks at once: shape (simulations, days)
        # Reuse array for simulation paths to minimize allocations
        simulation_paths = np.random.normal(0, adjusted_vol, (simulations, days))

        # Cumulative sum of log returns (In-place optimization)
        simulation_paths += final_daily_drift
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
            "lower_80": np.round(lower_80, 2).tolist(),
            "metadata": {
                "base_drift": base_drift,
                "adjusted_drift": final_daily_drift,
                "base_vol": daily_vol,
                "adjusted_vol": adjusted_vol,
                "sentiment_impact": view_drift_adjustment,
                "conviction_score": conviction_score
            }
        }

# Singleton for ease of import
forecasting_engine = ForecastingEngine()
