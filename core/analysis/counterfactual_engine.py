# core/analysis/counterfactual_engine.py

import pandas as pd
from dowhy import CausalModel

class CounterfactualEngine:
    """
    A module for performing counterfactual reasoning.
    """

    def __init__(self, data: pd.DataFrame, model: CausalModel):
        """
        Initializes the CounterfactualEngine.

        Args:
            data: The data to use for the analysis.
            model: The causal model.
        """
        self.data = data
        self.model = model

    def estimate_effect(self, treatment: str, outcome: str, method_name: str = "backdoor.linear_regression"):
        """
        Estimates the causal effect of a treatment on an outcome.

        Args:
            treatment: The name of the treatment variable.
            outcome: The name of the outcome variable.
            method_name: The name of the estimation method to use.

        Returns:
            The estimated causal effect.
        """
        identified_estimand = self.model.identify_effect()
        causal_estimate = self.model.estimate_effect(identified_estimand,
                                                     method_name=method_name,
                                                     target_units="ate")
        return causal_estimate
