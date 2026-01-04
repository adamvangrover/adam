# 🧠 Prompt Authoring Guide

This guide explains **how to write, organize, test, and maintain prompts** in the `adam` codebase.

Prompts are a **core abstraction** in `adam`: they encode intent, constraints, and expected behavior for agents while remaining **data-driven and model-agnostic**.

The system currently uses the **Adam Open Prompt Library (AOPL-v1.0)** standard, which relies primarily on Markdown files for readability and portability, backed by a **Prompt-as-Code** loading engine (`core.prompting`).

---

## 1. What Is a Prompt in `adam`?

In `adam`, a prompt is:

*   A **template** used by an agent to guide an LLM’s behavior.
*   Parameterized with runtime context (input, memory, tools, goals).
*   Stored as a **file** (Markdown, JSON, or YAML) in the `prompt_library`.

Prompts are typically stored under:

```
prompt_library/AOPL-v1.0/
```

They are consumed by:
*   Agent implementations in `core/agents/`
*   The loading utility `core.prompting.loader.load_prompt`

### Prompt Types

| Type                  | Purpose                                   | File Standard |
| --------------------- | ----------------------------------------- | ------------- |
| **System prompt**     | Sets role, tone, and global constraints   | Markdown (`.md`) |
| **Task prompt**       | Describes the specific task to perform    | Markdown (`.md`) |
| **Configuration**     | Defines strict schema and parameters      | YAML (`.yaml`) or JSON |

---

## 2. Prompt Structure & Conventions (AOPL-v1.0)

The **Adam Open Prompt Library (AOPL)** organizes prompts by **intent**.

### Directory Structure

Recommended structure under `prompt_library/AOPL-v1.0/`:

```
prompt_library/AOPL-v1.0/
  ├── learning/                 # Prompts for few-shot examples (e.g., `few_shot_qa.md`)
  ├── professional_outcomes/    # Prompts for final work products (e.g., `investment_memo.md`)
  ├── simulation/               # Prompts for scenarios (e.g., `crisis_simulation.md`)
  └── system_architecture/      # Prompts for core personas (e.g., `adam_v23_5_apex_architect.md`)
```

**Rules:**

*   **One prompt per file**: Use lowercase, descriptive filenames.
*   **Group by intent**: Organize by what the prompt *does* (e.g., `simulation`), not just which agent uses it.
*   **Format**: Markdown is preferred for human-readable text; JSON/YAML for structured config.

---

## 3. Writing High-Quality Prompts

### Core Principles

✅ **Be explicit**: Ambiguity is the enemy.
✅ **Constrain outputs**: Specify the exact format (JSON, Bullet points, etc.).
✅ **Be model-agnostic**: Avoid relying on hidden internal state of specific models.
✅ **Prefer instructions**: Modern models follow instructions well; use examples primarily for complex nuances.

### Template Variables

Prompts may include placeholders rendered at runtime. The `Prompt-as-Code` engine supports **Jinja2** syntax (`{{ variable }}`) and standard Python string formatting (`{variable}`).

**Example (Markdown):**

```markdown
# Role
You are a Senior Credit Analyst.

# Task
Analyze the creditworthiness of **{{ company_name }}**.

# Context
{{ market_context }}

# Instructions
1. Assess leverage ratios.
2. Identify key risks.
3. Provide a rating.
```

### Output Expectations

Always specify:
*   Output format (e.g., "Strict JSON").
*   Allowed/disallowed content.

**Example:**

```text
Output Format:
Return ONLY valid JSON. Do not include markdown fencing (```json).
{
  "rating": "string",
  "rationale": "string",
  "risks": ["string"]
}
```

---

## 4. Using Prompts in Code

The `core.prompting` module provides utilities to load prompts safely.

**Example Usage:**

```python
from core.prompting.loader import load_prompt

# Loads 'prompt_library/AOPL-v1.0/professional_outcomes/investment_memo.md'
# (The loader searches recursively)
prompt_template = load_prompt("investment_memo")

# Render with context
rendered_prompt = prompt_template.format(
    company_name="Acme Corp",
    market_context="Bull market..."
)

response = agent.run(rendered_prompt)
```

---

## 5. Advanced: Prompt-as-Code (YAML)

For complex prompts requiring metadata, versioning, or strict schema validation, you can use the **YAML** format supported by `BasePromptPlugin`.

**File: `prompt_library/AOPL-v1.0/my_complex_prompt.yaml`**

```yaml
prompt_id: "credit_risk_v2"
version: "2.0.0"
model_config:
  temperature: 0.1
system_template: |
  You are a risk engine.
user_template: |
  Analyze {{ company }}.
```

This is loaded via `BasePromptPlugin.from_yaml` and allows for pre-validation of inputs.

---

## 6. Testing Prompts

### Manual Testing
*   Run the prompt with minimal, maximal, and invalid inputs.
*   Verify the output format matches expectations.

### Automated Testing
*   Use `tests/test_prompt_framework.py` as a reference.
*   Create unit tests that load the prompt and verify it renders without errors.

---

## 7. Prompt Author Checklist

Before submitting a prompt:

*   [ ] **Clear Role & Task**: Does the prompt say *who* it is and *what* to do?
*   [ ] **Explicit Output**: Is the output format defined (JSON, etc.)?
*   [ ] **Variables**: Are all `{{ placeholders }}` known and provided by the code?
*   [ ] **File Location**: Is it in the correct `AOPL-v1.0` subdirectory?
*   [ ] **Tested**: Have you manually verified it produces good results?

---

## 📌 Summary

Prompts in `adam` are first-class artifacts.
*   **Write** in Markdown for clarity.
*   **Organize** in `AOPL-v1.0` by intent.
*   **Load** via `core.prompting`.
*   **Test** for reliability.
