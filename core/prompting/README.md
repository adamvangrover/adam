# Prompt-as-Code (PaC) Framework

This module provides a robust, type-safe framework for managing LLM prompts as code. It leverages Pydantic for strict input/output validation and Jinja2 for flexible template rendering.

## Core Concepts

*   **Plugin Architecture**: Every prompt is a Python class (`BasePromptPlugin`) that encapsulates:
    *   **Metadata**: Version, author, model config.
    *   **Input Schema**: Defines the variables injected into the prompt.
    *   **Template Logic**: Jinja2 templates (User/System separation supported).
    *   **Output Schema**: Defines the expected structure of the LLM response.
*   **Type Safety**: Inputs are validated *before* rendering. Outputs are parsed and validated *after* execution.
*   **Auditability**: Built-in logging ensures every interaction is traceable.

## Usage

### 1. Define a Plugin

```python
from core.prompting import BasePromptPlugin, PromptMetadata
from pydantic import BaseModel

class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str
    score: float

class SentimentPlugin(BasePromptPlugin[SentimentOutput]):
    def get_input_schema(self):
        return SentimentInput

    def get_output_schema(self):
        return SentimentOutput
```

### 2. Instantiate and Run

```python
metadata = PromptMetadata(
    prompt_id="sentiment_v1",
    author="jdoe",
    model_config={"temperature": 0.0}
)
plugin = SentimentPlugin(
    metadata,
    system_template="You are a sentiment analyzer.",
    user_template="Analyze: {{ text }}"
)

# Render for Chat API
messages = plugin.render_messages({"text": "I love this!"})
# -> [{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]

# Parse Response
result = plugin.parse_response('{"sentiment": "positive", "score": 0.9}')
print(result.sentiment)
```

### 3. Loading from Configuration

You can separate code from content by defining templates in YAML:

```yaml
# sentiment.yaml
prompt_id: "sentiment_v1"
version: "1.0.0"
template_body: "Analyze: {{ text }}"
model_config:
  temperature: 0.1
```

```python
plugin = SentimentPlugin.from_yaml("sentiment.yaml")
```

## Prompt Registry

Use `PromptRegistry` to manage multiple plugins.

```python
from core.prompting.registry import PromptRegistry
PromptRegistry.register(SentimentPlugin)
```
