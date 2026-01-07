# Adam Prompt Library (AOPL)

Welcome to the **Adam Operational Prompt Library (AOPL)**. This is the central cortex of the Adam system, containing the instructions, cognitive architectures, and functional templates that drive the behavior of our autonomous agents.

> **"Code defines the body; Prompts define the mind."**

## 🗂️ Library Structure

We follow a strict hierarchical structure to ensure prompts are modular, reusable, and versioned.

### 1. `AOPL-v1.0/` (Core Brain)
The foundational prompt sets for the v23 "Adaptive System".
*   **`system_architecture/`**: High-level meta-prompts.
    *   `AWO_System_Prompt.md`: Defines the "Architect" persona and meta-cognition.
    *   `MetaOrchestrator.md`: Routing logic for the central brain.
*   **`professional_outcomes/`**: Domain-specific expert personas.
    *   `credit_analysis.md`: Instructions for Shared National Credit (SNC) analysis.
    *   `market_analysis.md`: Guidelines for macro-economic trend spotting.
    *   `esg_analysis.md`: Directives for Environmental, Social, and Governance scoring.
*   **`learning/`**: Prompts for autonomous self-improvement.
    *   `reflection.md`: Instructions for the `ReflectorAgent` to critique its own work.
    *   `few_shot_examples.md`: A library of "Gold Standard" Q&A pairs for in-context learning.

### 2. Specialized Modules
*   **`risk_architect_agent/`**: Deep-dive prompts for the Vertical Risk Agent (SNCs, Covenants, Monte Carlo).

### 3. Root Files
*   **`Adam_v23.5_System_Prompt.md`**: **The Master Prompt**. This is the entry point for the "Apex Architect" model. It combines system instructions, personality vectors (HNASP), and tool definitions.

---

## 🚀 Usage Guide

### Loading Prompts in Code
Adam uses a dynamic loader to fetch prompts. Do not hardcode strings in Python files.

```python
from core.utils.prompt_utils import load_prompt

# 1. Load the Master System Prompt
system_prompt_template = load_prompt("Adam_v23.5_System_Prompt.md")

# 2. Render with Jinja2 Variables
rendered_prompt = system_prompt_template.render(
    user_query="Analyze Apple's debt structure",
    current_date="2023-10-27",
    security_context="INTERNAL_ONLY"
)

# 3. Pass to Agent
agent.execute(rendered_prompt)
```

### Dynamic Variable Injection
We use **Jinja2** templating. Common variables include:
*   `{{ user_query }}`: The raw input from the user.
*   `{{ context }}`: A JSON string or dict containing retrieved knowledge (RAG).
*   `{{ tools }}`: A list of available tools/functions.
*   `{{ persona }}`: The specific role the agent should adopt (e.g., "Skeptical Risk Officer").

---

## ✍️ Prompt Engineering Guidelines

When contributing to this library, adhere to the **"Chain of Thought" (CoT)** and **"Role-Task-Constraint"** principles.

### 1. Define the Persona
Every prompt must start by defining WHO the AI is.
> *Example:* "You are the Chief Risk Officer of a Tier-1 Investment Bank. You are skeptical, precise, and obsessed with downside protection."

### 2. Define the Task
Be explicit about the input and the expected output format.
> *Example:* "Input: A 10-K filing. Output: A JSON object containing 'EBITDA', 'Total Debt', and 'Leverage Ratio'."

### 3. Apply Constraints
Tell the model what NOT to do.
> *Example:* "Do not hallucinate data. If a metric is missing, write 'N/A'. Do not use markdown bolding in the JSON output."

### 4. Use XML Tags for Structure
We use XML tags to compartmentalize prompt sections for better LLM adherence.
```xml
<context>
{{ context }}
</context>

<instructions>
1. Analyze the context.
2. Calculate ratios.
</instructions>

<output_format>
JSON only.
</output_format>
```

---

## 🛠️ Contribution Workflow

1.  **Create**: Draft your prompt in a `.md` file.
2.  **Categorize**: Place it in `AOPL-v1.0/professional_outcomes/` (if it's a task) or `system_architecture/` (if it's a behavior).
3.  **Test**: Run it against the `tests/test_prompt_framework.py` suite to ensure variables render correctly.
4.  **PR**: Submit your PR with the tag `[PROMPT]`.

---

## 📚 Reference: Available Personas

| Persona ID | Role | File Path |
| :--- | :--- | :--- |
| `APEX_ARCHITECT` | System Controller | `Adam_v23.5_System_Prompt.md` |
| `CREDIT_ANALYST` | SNC & Debt Specialist | `AOPL-v1.0/professional_outcomes/credit_analysis.md` |
| `ESG_SCORER` | Sustainability Auditor | `AOPL-v1.0/professional_outcomes/esg_analysis.md` |
| `REFLECTOR` | Critic & improver | `AOPL-v1.0/learning/reflection.md` |

