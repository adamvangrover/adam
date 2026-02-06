# Core Components & Agent Guidelines

This directory (`core/`) contains the "System 2" brain of Adam.

## ⚠️ Critical Engineering Standards

Agents working in this directory must adhere to strict standards to ensure the stability of the financial reasoning engine.

### 1. Strict Typing
*   **Pydantic Everywhere:** All data structures, especially Agent State and Tool Inputs/Outputs, must be defined using `pydantic.BaseModel`.
*   **Type Hints:** All functions must have type hints for arguments and return values.

```python
from pydantic import BaseModel, Field

class StockQuery(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL)")
    depth: str = Field("standard", description="Analysis depth: standard or deep_dive")
```

### 2. No Hallucinations (Grounding)
*   **Source Citation:** Every analytical claim made by an agent must cite a source (e.g., "According to the 2023 10-K...").
*   **Confidence Scores:** When uncertain, agents must output a low confidence score (0-100) or flag the data as "Unverified".

### 3. Graceful Degradation
*   **Dependency Checks:** If a heavy library (e.g., `torch`, `spacy`) is missing, the module must not crash on import. Use `try/except ImportError` blocks and provide a mocked or simplified fallback.
*   **API Failures:** If an external API (OpenAI, FMP) fails, the agent should return a structured error state, not raise an unhandled exception.

### 4. LangGraph State Management
*   The `core/engine` uses `langgraph`.
*   All graph nodes must accept `state` as input and return a dictionary of state updates.
*   Do not mutate the state in place; return the diff.

## Directory Structure

*   **`agents/`:** Specialized workers (Fundamental, Risk, Legal).
*   **`engine/`:** The orchestrator and graph definitions.
*   **`data_processing/`:** The "Universal Ingestor" (ETL).

## Testing
Before submitting changes to `core/`:
1.  Run the relevant unit tests in `tests/`.
2.  Run `python scripts/run_adam.py --query "test query"` to ensure the graph compiles and runs.
