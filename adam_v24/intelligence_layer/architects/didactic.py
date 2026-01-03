import ast
import inspect
from typing import Callable

class DidacticArchitect:
    """
    Meta-Agent III: The Didactic Architect.
    Ensures interpretability via automated documentation and interactive tutorials.
    """

    def check_documentation_drift(self, func: Callable, docstring: str) -> bool:
        """
        Parses function signature and compares it with docstring.
        """
        sig = inspect.signature(func)
        params = sig.parameters.keys()

        # Simple heuristic: Check if all parameters are mentioned in docstring
        missing_params = [p for p in params if p not in docstring and p != "self"]

        if missing_params:
            print(f"Didactic Alert: Doc drift detected in {func.__name__}. Missing params: {missing_params}")
            return True
        return False

    def generate_marimo_tutorial(self, scenario_name: str) -> str:
        """
        Generates a Marimo notebook code string for a given scenario.
        """
        notebook_content = f"""
import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __(mo):
    mo.md("# Scenario: {scenario_name}")
    return

@app.cell
def __(mo):
    risk_tolerance = mo.ui.slider(0, 100, label="Risk Tolerance")
    risk_tolerance
    return risk_tolerance

@app.cell
def __(risk_tolerance):
    # Simulated reaction
    exposure = risk_tolerance.value * 1.5
    print(f"Calculated Market Exposure: {{exposure}}M USD")
    return exposure

if __name__ == "__main__":
    app.run()
"""
        return notebook_content
