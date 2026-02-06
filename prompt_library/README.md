# Adam Operational Prompt Library (AOPL) v26.0

Welcome to the **Adam Operational Prompt Library**. This is the central cortex of the Adam system, containing the instructions, cognitive architectures, and functional templates that drive the behavior of our autonomous agents.

> **"Code defines the body; Prompts define the mind."**

---

## 🗂️ Library Structure

We follow a strict hierarchical structure to ensure prompts are modular, reusable, and versioned.

### 1. Root Files (Master Prompts)
*   `Adam_v26.0_System_Prompt.md`: **The Master Prompt**. This is the entry point for the "Apex Architect" model. It combines system instructions, personality vectors (HNASP), and tool definitions.

### 2. `AOPL-v2.0/` (Current Standard)
The foundational prompt sets for the v26 "Neuro-Symbolic Sovereign".

| Directory | Description |
| :--- | :--- |
| `system_architecture/` | High-level meta-prompts (Meta Orchestrator, Planner). |
| `professional_outcomes/` | Domain-specific expert personas (Credit Analyst, Market Watcher). |
| `learning/` | Prompts for autonomous self-improvement and reflection. |

### 3. `AOPL-v1.0/` (Legacy)
Retained for backward compatibility. Do not use for new agents.

---

## 🚀 Usage Guide

### Loading Prompts in Code
Adam uses a dynamic loader to fetch prompts. **Do not hardcode strings in Python files.**

```python
from core.utils.prompt_utils import load_prompt

# 1. Load the Master System Prompt
system_prompt_template = load_prompt("Adam_v26.0_System_Prompt.md")

# 2. Render with Jinja2 Variables
rendered_prompt = system_prompt_template.render(
    user_query="Analyze Apple's debt structure",
    current_date="2026-03-15",
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

---

## 🛠️ Contribution Workflow

1.  **Draft**: Create a new `.md` file in the appropriate `AOPL-v2.0/` subdirectory.
2.  **Test**: Run it against the `tests/test_prompt_framework.py` suite.
3.  **PR**: Submit your PR with the tag `[PROMPT]`.
