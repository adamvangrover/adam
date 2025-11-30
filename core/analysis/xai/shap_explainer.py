# core/analysis/xai/shap_explainer.py

import shap

class SHAPExplainer:
    """
    An implementation of the SHAP (SHapley Additive exPlanations) algorithm.
    """

    def __init__(self, model, data):
        """
        Initializes the SHAPExplainer.

        Args:
            model: The model to explain.
            data: The data to use for the explanation.
        """
        self.model = model
        self.data = data
        self.explainer = shap.KernelExplainer(self.model.predict, self.data)

    def explain(self, instance):
        """
        Generates an explanation for a single instance.

        Args:
            instance: The instance to explain.

        Returns:
            The SHAP values for the instance.
        """
        return self.explainer.shap_values(instance)
