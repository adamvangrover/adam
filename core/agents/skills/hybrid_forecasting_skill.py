# core/agents/skills/hybrid_forecasting_skill.py

from semantic_kernel.skill_definition import sk_function, sk_description
from core.analysis.forecasting.hybrid_forecaster import HybridForecaster
import pandas as pd

# Developer Note: This skill provides an interface to the hybrid forecasting
# capabilities of the system. Agents can use this skill to generate
-
# robust time-series forecasts.

class HybridForecastingSkill:
    """
    A Semantic Kernel skill for hybrid time-series forecasting.
    """

    def __init__(self):
        self.forecaster = HybridForecaster()

    @sk_function(
        description="Generates a time-series forecast.",
        name="forecast",
    )
    def forecast(self, data: list, steps: int) -> list:
        """
        Generates a forecast for the given time series data.

        Args:
            data: A list of numerical data points.
            steps: The number of steps to forecast into the future.

        Returns:
            A list containing the forecasted values.
        """
        series = pd.Series(data)
        return self.forecaster.forecast(series, steps)
