# Skeleton & Inject Workflow Library

## Overview

The "Skeleton & Inject" workflow is a **Prompt-as-Code (PaC)** module designed for high-precision financial analysis generation. It solves the "Hallucination Problem" by enforcing a strict separation between narrative generation (Phase 1) and data injection (Phase 2).

## Core Philosophy

1.  **Narrative Skeleton (Phase 1):** The LLM writes the *story* but is forbidden from writing numbers. It uses placeholders like `{{REVENUE_CURRENT}}`.
2.  **Data Layer (Middleware):** A Python layer extracts these placeholders and fetches verified data from a trusted source (API, SQL, or Vector DB).
3.  **Synthesis & Audit (Phase 2):** The LLM acts as an editor, injecting the true numbers and adjusting adjectives ("robust" -> "weak") to match the reality.
4.  **Critique (Phase 3):** A final "Senior Credit Officer" agent reviews the output for logic and tone.

## Usage

```python
from core.prompting.workflows.skeleton_inject import SkeletonInjectWorkflow, JSONFileFetcher

# 1. Setup Data Source
fetcher = JSONFileFetcher("path/to/financial_data.json")

# 2. Initialize Workflow
workflow = SkeletonInjectWorkflow(
    llm_client=my_llm_client,
    data_fetcher=fetcher,
    tone="Bearish" # Optional: Adjusts system prompt persona
)

# 3. Run
context = "Earnings Call Transcript..."
result = workflow.run(context)

print(result.final_text)
print(result.critique) # Review SCO feedback
```

## Adding New Prompts

Prompts are stored in `prompt_library/AOPL-v1.0/analyst_os/`. They use Markdown with YAML Frontmatter.

To add a new phase or modify existing ones:
1. Create/Edit the `.md` file in the library.
2. The `PromptLoader` will automatically pick up changes without code deploys.

## Data Fetchers

*   **MockDataFetcher:** Uses hardcoded test data.
*   **JSONFileFetcher:** Reads a flat JSON file (useful for demos).
*   **Custom:** Implement the `DataFetcher` interface to connect to SQL/Bloomberg.
