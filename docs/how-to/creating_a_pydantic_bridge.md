# How-To: Creating a Pydantic Bridge

This guide demonstrates how to enforce type safety when moving data from stochastic System 1 agents to deterministic System 2 Rust kernels.

## The PDIL Bridge
The Probabilistic-to-Deterministic Integration Layer (PDIL) requires strict data contracts. We use Pydantic models to ensure outputs from LLM agents meet the expected schemas before entering the execution layer.

## Step 1: Define the Schema
Define your strictly typed Pydantic model in the `core_types.py` or relevant schema file.

```python
from pydantic import BaseModel, Field

class ProvenanceHeader(BaseModel):
    source_uri: str
    timestamp: str
    hash: str

class AgentOutput(BaseModel):
    conviction_score: float = Field(..., ge=0.0, le=1.0)
    provenance_trace: ProvenanceHeader = Field(...)
```

*Note: As per our conventions, the `provenance_trace` must be explicitly defined and cannot be `Optional` or use a dummy default.*

## Step 2: Validate Agent Outputs
When the semantic engine returns data, validate it using the schema.

```python
output_data = sentiment_engine.process(text)
validated_data = AgentOutput.model_validate(output_data)
```

## Step 3: Pass to Execution
The validated JSON string representation of `validated_data` can safely be handed off via the API gateway to the Rust pricing kernels, confident that all required fields and type constraints are met.
