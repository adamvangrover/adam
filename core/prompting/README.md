# Prompt-as-Code & Advanced Reasoning Framework

## Overview
This directory contains the **Prompt-as-Code** infrastructure (`BasePromptPlugin`) and the implementation of advanced reasoning strategies inspired by DeepMind and Google Research.

## Core Architecture
The framework treats prompts as software artifacts:
*   **Versioned:** Prompts have IDs, versions, and authors (`PromptMetadata`).
*   **Typed:** Inputs and Outputs are validated using Pydantic schemas.
*   **Composable:** Prompts can be chained or composed of smaller modules.

## Advanced Reasoning Modules (`core/prompting/advanced_reasoning.py`)

### 1. Self-Discover Framework (`SelfDiscoverPrompt`)
*   **Concept:** Based on the paper *"Self-Discover: Large Language Models Self-Compose Reasoning Structures"* by DeepMind.
*   **Process:** Instead of using a fixed prompt (like "Think step-by-step"), the model first **selects** useful reasoning modules (e.g., "Critical Thinking", "Decomposition"), **adapts** them to the specific task, and then **implements** a plan.
*   **Usage:**
    ```python
    plugin = SelfDiscoverPrompt()
    structure = plugin.render({"task_description": "Analyze AAPL", "context": "..."})
    # Returns a plan like: "1. Decompose revenue... 2. Critically analyze risks..."
    ```

### 2. Chain-of-Verification (`ChainOfVerificationPrompt`)
*   **Concept:** Based on *"Chain-of-Verification Reduces Hallucination in Large Language Models"*.
*   **Process:**
    1.  **Draft:** Generate an initial response.
    2.  **Plan:** Identify facts that need verification.
    3.  **Execute:** Answer verification questions (using tools or internal knowledge).
    4.  **Revise:** Produce a final, verified response.
*   **Usage:** Useful for high-stakes assertions (e.g., specific revenue numbers or dates).

## Extending
To add a new prompt strategy:
1.  Inherit from `BasePromptPlugin`.
2.  Define `InputSchema` and `OutputSchema` (Pydantic).
3.  Implement the `template` or logic in `__init__`.
