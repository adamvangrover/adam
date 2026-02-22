# Operationalizing Financial Truth: The TAO-CoT Framework

## Overview
This document details the implementation of the "Financial Truth" reasoning engine within the Adam v23.5 architecture. This module is designed to mitigate the "Epistemological Crisis" in financial AI—where probabilistic models fail to deliver deterministic accuracy required for auditing—by operationalizing the **TAO (Task, Analysis, Output)** framework.

## The Epistemological Crisis
Financial analysis requires:
1.  **Closed World Adherence:** Answers must come *only* from the provided documents (10-Ks, 10-Qs), ignoring the model's stale pre-training data.
2.  **Auditability:** Every number must be traceable to a specific sentence or table row.
3.  **Numerical Precision:** "1.2 billion" is not "1.2 million".

Generalist models fail this test ~81% of the time on benchmarks like **FinanceBench**.

## The Solution: TAO Framework
We have implemented a "System 2" reasoning prompt that forces the model to slow down and verify its work.

### 1. Task (The Closed World Constraint)
*   **Zero External Knowledge:** We explicitly disable access to outside facts.
*   **Refusal is Accuracy:** The model is trained to say "Information not available" rather than hallucinate.

### 2. Analysis (The Reasoning Engine)
The prompt enforces a `<thinking>` block before the answer.
*   **Unit Scan:** Checks for "millions vs billions".
*   **Needle in Haystack:** Locates the specific row.
*   **Math Check:** Shows the formula used (e.g., `(Cash + Marketable Securities) / Current Liabilities`).

### 3. Output (The Information Triplet)
The output is structured for machine parsing:
*   **Answer:** The direct response.
*   **Evidence:** The verbatim quote or table row.
*   **Logic:** The reasoning trace.

## Implementation Details

### Artifacts
*   **Prompt Template:** `prompt_library/AOPL-v1.0/professional_outcomes/LIB-PRO-009_financial_truth_tao.md`
*   **Python Plugin:** `core/prompting/plugins/financial_truth_plugin.py`
*   **Input Schema:** `core/schemas/financial_truth.py`

### Usage Example
```python
from core.prompting.registry import PromptRegistry
from core.prompting.loader import PromptLoader

# Load the plugin
PluginClass = PromptRegistry.get("FinancialTruthPlugin")
template_content = PromptLoader.get("AOPL-v1.0/professional_outcomes/LIB-PRO-009_financial_truth_tao")

plugin = PluginClass(
    metadata=...,
    user_template=template_content
)

# Execute
response = plugin.render_messages({
    "context": "Net Income: $500M...",
    "question": "What is the Net Income?"
})
# Send `response` to LLM...
# Parse result
result = plugin.parse_response(llm_output)
print(result.evidence) # "Net Income: $500M"
```

## Performance Metrics (Expected)
Based on FinanceBench analysis, this "System 2" approach (CoT + RAG) is expected to reduce the failure rate from ~81% (standard RAG) to <25%.

## References
*   **FinanceBench:** A New Benchmark for Financial Question Answering.
*   **TAO:** Task, Analysis, Output framework for prompt engineering.
