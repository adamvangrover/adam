# Adam Finance

## Overview
This module (`adam_finance`) houses deterministic mathematical execution, containing purely quantitative financial algorithms, such as Value at Risk (VaR) and Structured Notes Calculator (SNC) functions.

## Objectives
- Host all pure financial mathematical functions and logic algorithms.
- Maintain isolation from probabilistic Systems 1 logic, ensuring consistent and completely deterministic mathematical calculations.
- Remain completely free of any dependency on `langchain`, `semantic_kernel`, or any other LLM-execution toolsets.

## Future Plans
- Expand deterministic models to cover advanced Greek calculations for options.
- Move out financial modules currently mixed in the `core/engine/` layer to centralize here.
- Add robust Monte Carlo pipelines for deterministic execution sequences.
