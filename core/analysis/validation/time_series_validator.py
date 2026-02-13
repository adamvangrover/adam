import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from core.engine.forecasting_engine import forecasting_engine

class TimeSeriesValidator:
    """
    Validates forecasting models by running rolling backtests and calculating
    metrics weighted by conviction.
    """

    def __init__(self):
        pass

    def rolling_window_backtest(self,
                                symbol: str,
                                history: List[Dict[str, Any]],
                                horizon_days: int = 30,
                                step_days: int = 7,
                                start_idx: int = 100) -> List[Dict[str, Any]]:
        """
        Runs a backtest over the historical data.

        Args:
            symbol: Ticker symbol.
            history: Full list of historical price records (dicts with 'close', 'date').
            horizon_days: How far ahead to forecast.
            step_days: How many days to move the window forward each iteration.
            start_idx: Minimum history length required before first forecast.

        Returns:
            List of result records containing forecast vs actuals and metrics.
        """
        results = []
        n = len(history)

        # Iterate through history
        for i in range(start_idx, n - horizon_days, step_days):
            window_history = history[:i]
            actual_future = history[i:i+horizon_days]
            actual_price_at_horizon = actual_future[-1]['close']

            # Simulate "Agent Signals" for this point in time
            # In a real system, we'd look up historical agent logs.
            # Here we mock it based on simple momentum (just to test the mechanism)
            recent_return = (window_history[-1]['close'] / window_history[-30]['close']) - 1
            sentiment = 1.0 if recent_return > 0 else -1.0
            conviction = min(1.0, abs(recent_return) * 5) # Higher momentum -> Higher conviction

            # Generate Forecast
            forecast = forecasting_engine.generate_forecast(
                symbol,
                window_history,
                days=horizon_days,
                sentiment_score=sentiment,
                conviction_score=conviction
            )

            predicted_mean = forecast['mean'][-1]

            # Calculate Error
            error = predicted_mean - actual_price_at_horizon
            abs_error = abs(error)
            sq_error = error ** 2

            # Calculate Weighted Error (Conviction Penalty)
            # If we had high conviction and missed, the penalty is higher.
            # If we had low conviction, the penalty is lower.
            weighted_penalty = abs_error * (1 + conviction)

            results.append({
                "date": window_history[-1]['date'],
                "horizon_date": actual_future[-1]['date'],
                "actual": actual_price_at_horizon,
                "predicted": predicted_mean,
                "error": error,
                "abs_error": abs_error,
                "sentiment": sentiment,
                "conviction": conviction,
                "weighted_penalty": weighted_penalty
            })

        return results

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregates backtest results into summary metrics.
        """
        if not results:
            return {}

        df = pd.DataFrame(results)

        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        mean_weighted_penalty = df['weighted_penalty'].mean()

        # Directional Accuracy
        # (Did the sign of the forecast change match the sign of the actual change?)
        # For simplicity, let's just assume prediction direction vs current price
        # This requires passing current price, but let's approximate.

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Conviction_Weighted_Penalty": round(mean_weighted_penalty, 4),
            "Samples": len(df)
        }

# Singleton
time_series_validator = TimeSeriesValidator()
