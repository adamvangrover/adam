# Explainable AI (XAI)

## Overview

Adam v22.0 makes its reasoning processes more transparent and understandable to human users through the use of Explainable AI (XAI) techniques.

## SHAP (SHapley Additive exPlanations)

Adam uses the SHAP algorithm to generate explanations for the outputs of its machine learning models. SHAP is a game theory-based approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

## XAISkill

The `XAISkill` is a Semantic Kernel skill that allows agents to generate explanations for their recommendations. The skill provides functions for calling the `SHAPExplainer` and formatting the output in a human-readable way.

## Visualizations

SHAP explanations can be visualized to provide a clear and intuitive understanding of the model's predictions.
