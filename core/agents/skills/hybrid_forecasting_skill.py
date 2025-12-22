# core/agents/skills/hybrid_forecasting_skill.py

from semantic_kernel.skill_definition import sk_function, sk_description
from core.analysis.forecasting.hybrid_model import HybridModel as HybridForecaster
import pandas as pd
import json

# Developer Note: This skill provides an interface to the hybrid forecasting
# capabilities of the system. Agents can use this skill to generate
# robust time-series forecasts and apply qualitative adjustments.

class HybridForecastingSkill:
    """
    A Semantic Kernel skill for hybrid time-series forecasting (ARIMA + LSTM)
    with qualitative risk adjustment.
    """

    def __init__(self):
        # In a real scenario, we might load a pre-trained model or fit on demand
        # Default parameters from the roadmap (v23.5)
        self.forecaster = HybridForecaster(arima_order=(5, 1, 0), lstm_units=50)
        self.is_fitted = False

    @sk_function(
        description="Generates a time-series forecast.",
        name="forecast",
    )
    def forecast(self, data_json: str, steps: int) -> str:
        """
        Generates a forecast for the given time series data.

        Args:
            data_json: A JSON string list of numerical data points.
            steps: The number of steps to forecast into the future.

        Returns:
            A JSON string containing the forecasted values.
        """
        try:
            data = json.loads(data_json)
            series = pd.Series(data)

            # For the purpose of this skill, we fit on the fly if not fitted
            # In production, this would use a persisted model
            self.forecaster.fit(series)
            self.is_fitted = True

            forecast_series = self.forecaster.predict(steps)
            return json.dumps(forecast_series.tolist())
        except Exception as e:
            return f"Error in forecasting: {str(e)}"

    @sk_function(
        description="Adjusts a quantitative forecast based on qualitative risk factors (Hybrid adjustment).",
        name="adjust_forecast",
    )
    def adjust_forecast(self, forecast_json: str, risk_factor: float, sentiment_score: float) -> str:
        """
        Adjusts a forecast based on qualitative inputs (e.g., news sentiment, geopolitical risk).

        Roadmap v23.5: "The skill then adjusts the forecast downward, integrating 'soft' information into 'hard' numbers."

        Args:
            forecast_json: The base quantitative forecast (JSON list).
            risk_factor: A scaler (0.0 to 1.0) representing external risk (1.0 = high risk).
            sentiment_score: A scaler (-1.0 to 1.0) representing market sentiment.

        Returns:
            A JSON string of the adjusted forecast.
        """
        try:
            forecast = json.loads(forecast_json)
            adjusted_forecast = []

            # Simple heuristic adjustment logic for v23.5 MVP
            # If sentiment is negative and risk is high, dampen the forecast
            adjustment_multiplier = 1.0

            if sentiment_score < -0.2:
                adjustment_multiplier -= 0.05 * abs(sentiment_score) # Downward pressure

            if risk_factor > 0.5:
                adjustment_multiplier -= 0.1 * (risk_factor - 0.5) # Additional penalty for high risk

            # "Fat-Tail" Simulation hook:
            # If risk is extreme (>0.8), we might model a shock (not implemented fully here, but placeholder)
            if risk_factor > 0.8:
                # Simulate a drop in the later steps
                pass

            for val in forecast:
                adjusted_forecast.append(val * adjustment_multiplier)

            return json.dumps(adjusted_forecast)
        except Exception as e:
            return f"Error in adjustment: {str(e)}"
