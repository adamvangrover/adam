# 🧠 Async Coding Swarm Prompt: Prompt Authoring Guide

## 📘 Goal

This guide serves as the definitive manual for creating, modifying, and managing prompts within the `adam` ecosystem. It is designed for both **Prompt Engineers** working with raw text templates and **Software Engineers** building robust "Prompt-as-Code" implementations.

In the `adam` repository, we treat prompts as software artifacts. They have versions, tests, schemas, and strictly defined inputs/outputs.

---

## 1. 📂 Prompt Architecture

Prompts in `adam` are stored in two primary ways:

1.  **Static Templates (`prompt_library/`)**: Paired `.md` and `.json` files used for standard tasks.
2.  **Prompt Plugins (`core/prompting/`)**: Python classes inheriting from `BasePromptPlugin` for complex, type-safe interactions.

### Directory Structure

```text
adam/
├── prompt_library/             # The central repository for static prompts
│   ├── AOPL-v1.0/              # Adam Open Prompt Library (New Standard)
│   ├── credit_analysis.md      # The Prompt Template (Jinja2)
│   ├── credit_analysis.json    # The Configuration (Model params, Schema)
│   ├── advanced/               # Advanced patterns (ToT, CoVe)
│   │   ├── tot_scenario.md
│   │   └── cove_fact_check.md
│   └── ...
└── core/
    └── prompting/              # The Engine
        ├── base_prompt_plugin.py  # Base class for Prompt-as-Code
        ├── loader.py              # Logic to load .md/.json pairs
        ├── plugins/               # Logic-heavy prompt implementations
        │   ├── tree_of_thoughts_plugin.py
        │   └── chain_of_verification_plugin.py
        └── registry.py            # Central registry for active prompts
```

---

## 2. 📝 The Anatomy of a Prompt

A standard prompt consists of two files sharing the same name.

### A. The Template (`.md`)

We use **Markdown** for structure and **Jinja2** for variable substitution.

**File:** `prompt_library/example_task.md`

```markdown
# Task Name & Version
**Version:** 1.0
**Role:** Senior Analyst
**Task:** {{task_description}}

---

## 1. Context
You are analyzing **{{entity_name}}**. The current market environment is described as:
{{market_context}}

## 2. Instructions
Perform the following steps:
1. Review the provided financial data.
2. Assess risks related to {{risk_factors}}.
3. Generate a structured output.

## 3. Output Format
Return your response in strict JSON format:
```json
{
  "analysis": "string",
  "score": number,
  "flags": ["string"]
}
```
```

### B. The Configuration (`.json`)

This file controls the LLM parameters and documents expected inputs.

**File:** `prompt_library/example_task.json`

```json
{
  "prompt_id": "example_task_v1",
  "description": "Analyzes an entity and provides a risk score.",
  "model_config": {
    "temperature": 0.3,
    "max_tokens": 1024,
    "stop_sequences": ["### End"]
  },
  "input_variables": [
    "task_description",
    "entity_name",
    "market_context",
    "risk_factors"
  ],
  "output_format": "json"
}
```

---

## 3. 🏗️ "Prompt-as-Code" Implementation

For production agents, we encourage using the **Prompt Plugin** pattern defined in `core/prompting/base_prompt_plugin.py`. This ensures type safety using Pydantic.

### Step 1: Define Inputs/Outputs

```python
from pydantic import BaseModel, Field

class AnalysisInput(BaseModel):
    entity_name: str
    financial_data: dict

class AnalysisOutput(BaseModel):
    risk_score: float = Field(..., ge=0, le=10)
    summary: str
```

### Step 2: Create the Plugin

```python
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

class RiskAnalysisPlugin(BasePromptPlugin[AnalysisOutput]):
    def get_input_schema(self):
        return AnalysisInput

    def get_output_schema(self):
        return AnalysisOutput
```

### Step 3: Usage

```python
# Instantiate
plugin = RiskAnalysisPlugin.from_yaml("config/risk_prompt_v1.yaml")

# Render
prompt_text = plugin.render({
    "entity_name": "Acme Corp",
    "financial_data": {"revenue": 1000}
})

# ... Send prompt_text to LLM ...

# Parse & Validate
result = plugin.parse_response(raw_llm_response)
print(result.risk_score) # Typed float
```

---

## 4. ✅ Writing High-Quality Prompts

### The "Good vs Bad" Checklist

| Feature | ❌ Bad Prompt | ✅ Good Prompt |
| :--- | :--- | :--- |
| **Context** | "Analyze this company." | "You are a Senior Credit Officer analyzing [Company] for a Leveraged Buyout." |
| **Constraints** | "Keep it short." | "Limit the executive summary to 3 sentences or 100 words." |
| **Input Data** | (Assumes data is known) | "Use the following JSON data as your **only** source of truth: {{financial_data}}" |
| **Output** | "Give me a JSON." | "Output **only** raw JSON. No markdown formatting. Use this schema: {...}" |
| **Thinking** | (Direct answer) | "First, list your assumptions. Second, calculate ratios. Finally, provide the verdict." |

### Best Practices

1.  **Be Explicit about Roles:** Start with "You are a [Role]..." to anchor the model's persona.
2.  **Use Delimiters:** When passing large data chunks, use XML tags or separators to help the model distinguish instructions from data.
    *   *Example:* "Analyze the text inside `<news_article>` tags."
3.  **One Shot / Few Shot:** Providing just one example of the desired input/output pair significantly improves adherence to schemas.
4.  **Chain of Thought (CoT):** Ask the model to "think step-by-step" before providing the final answer. This is crucial for math or logic tasks.

---

## 5. 🧠 Advanced Reasoning Patterns

This section details sophisticated prompting strategies used in the Adam ecosystem for high-stakes financial reasoning.

### A. Chain of Thought (CoT)

Instead of asking for a direct answer, CoT forces the model to externalize its reasoning steps.

**Template Pattern:**
```markdown
Q: {{question}}
A: Let's think step by step.
1. Identify the key financial metrics in the question.
2. Locate the relevant data points in the provided context.
3. Perform the necessary calculations (show your work).
4. Formulate the final answer.
```

### B. Tree of Thoughts (ToT)

For problems requiring exploration or planning (e.g., "What is the best restructuring strategy?"), ToT enables the model to branch out into multiple "thought paths" and self-evaluate.

**Pattern:**
1.  **Thought Generation:** "Propose 3 distinct strategic options for debt restructuring."
2.  **State Evaluation:** "Critique each option based on feasibility, cost, and stakeholder impact. Assign a score (0.0 - 1.0)."
3.  **Search:** Select the best path and expand (DFS or BFS).

*See `core/prompting/plugins/tree_of_thoughts_plugin.py` for implementation.*

### C. Chain of Verification (CoVe)

To reduce hallucinations, use a "Draft -> Verify -> Revise" loop.

**Pattern:**
1.  **Draft:** Generate an initial response.
2.  **Verify:** "List 3 factual claims made in the draft. For each, verify it against the source text."
3.  **Revise:** "Rewrite the draft, correcting any errors found during verification."

*See `core/prompting/plugins/chain_of_verification_plugin.py` for implementation.*

---

## 6. 🔗 Prompt Chaining Strategies

Complex workflows often require connecting multiple prompts.

### 1. Sequential Chain
The output of Prompt A becomes the input for Prompt B.
*   *Use Case:* Data Extraction -> Summary Generation.
*   *Best Practice:* Ensure Prompt A outputs structured data (JSON) so Prompt B can reliably ingest it.

### 2. Router Chain (Classifier)
An initial prompt classifies the intent and selects the next specialized prompt.
*   *Use Case:* User asks a question -> Router detects "Credit", "Market", or "Legal" -> Routes to specific Expert Agent.

### 3. Map-Reduce (Fan-Out / Fan-In)
*   **Map:** Apply the same prompt to chunks of a large document (e.g., "Summarize this page").
*   **Reduce:** Combine the chunk outputs into a final answer (e.g., "Synthesize these 50 page summaries into an executive briefing").

---

## 7. 🔄 Versioning & Iteration

*   **Never overwrite** a prompt that is currently in production without backwards compatibility.
*   **Versioning:** Append versions to filenames or IDs (e.g., `credit_analysis_v2`).
*   **Deprecation:** When deprecating, mark the JSON config with `"status": "deprecated"` (if supported) or move to a `deprecated/` subfolder.

---

## 8. 🧪 Testing Prompts

Prompts should be tested just like code.

### Manual Testing
Use the `PromptLoader` to render prompts with mock data and inspect the output.

```python
from core.prompting.loader import PromptLoader

loader = PromptLoader()
data = loader.get_full_prompt_system("credit_analysis")
print(data['template']) # Verify Jinja2 rendering works
```

### Automated Testing (Concept)

In your unit tests, you can assert that:
1.  The prompt renders without errors given valid inputs.
2.  The input variables in the JSON config match the Jinja2 variables in the MD file.

```python
def test_prompt_integrity():
    loader = PromptLoader()
    sys = loader.get_full_prompt_system("credit_analysis")

    template = sys['template']
    inputs = sys['config']['input_variables']

    for var in inputs:
        assert f"{{{{ {var} }}}}" in template or f"{{{{{var}}}}}" in template
```

---

## 9. 📚 Quick Reference

*   **New Prompt?** Create `prompt_library/my_task.md` and `prompt_library/my_task.json`.
*   **New Logic?** Implement `BasePromptPlugin` in `core/prompting/plugins/`.
*   **Formatting?** Use Markdown headers and clear sections.
*   **Variables?** Use `{{snake_case}}` Jinja2 syntax.
