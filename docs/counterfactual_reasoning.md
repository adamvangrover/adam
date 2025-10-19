# Counterfactual Reasoning

## Overview

Adam v22.0 can perform "what if" analysis by leveraging the causal models in the Knowledge Graph. This is made possible by the `CounterfactualEngine`, a module that uses the `dowhy` library to perform causal inference.

## How it Works

1.  An agent defines an intervention (e.g., "if the Fed had not raised interest rates") and an outcome variable.
2.  The agent invokes the `CounterfactualReasoningSkill`.
3.  The skill uses the `CounterfactualEngine` to estimate the causal effect of the intervention on the outcome.

## Assumptions and Limitations

The accuracy of the counterfactual reasoning depends on the quality of the underlying causal models. It is important to carefully validate the causal models before using them for counterfactual reasoning.
