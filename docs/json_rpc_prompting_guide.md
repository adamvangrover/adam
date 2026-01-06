# JSON-RPC Prompting Guide

This guide details how to use the **JSON-RPC Prompt Library** and **Adaptive Conviction** patterns within the Adam v23.5 architecture.

## Overview

The JSON-RPC Prompting framework is designed to solve the "Protocol Paradox" by enforcing strict JSON-RPC 2.0 schemas for tool execution while maintaining a metacognitive "Ambiguity Guardrail" for the agent.

## Core Components

1.  **Schemas** (`core/schemas/json_rpc.py`):
    *   `JsonRpcRequest`: Standard JSON-RPC 2.0 request object.
    *   `AdaptiveConvictionMetadata`: Metadata for conviction scoring and mode switching.

2.  **Library** (`core/prompting/json_rpc_library.py`):
    *   `RESEARCH_DEEP_DIVE`: Complex research orchestration.
    *   `SYNTHESIS_REPORT`: Citational summarization.
    *   `REASONING_CHAIN`: Step-by-step logic.
    *   `SNC_ANALYSIS`: Specialized financial risk analysis.

3.  **Plugin** (`core/prompting/plugins/json_rpc_plugin.py`):
    *   `JsonRpcPromptPlugin`: A `BasePromptPlugin` implementation that wraps the templates and enforces schemas.

## Usage Example

```python
from core.prompting.plugins.json_rpc_plugin import JsonRpcPromptPlugin
import json

# 1. Instantiate the Plugin
plugin = JsonRpcPromptPlugin.from_registry("RESEARCH_DEEP_DIVE")

# 2. Define Inputs
inputs = {
    "topic": "Impact of Quantum Computing on Credit Risk Models",
    "tools": json.dumps([
        {"name": "search_web", "description": "Search the internet..."},
        {"name": "search_academic", "description": "Search papers..."}
    ])
}

# 3. Render Prompt
prompt_text = plugin.render(inputs)

# 4. (Mock) Simulate LLM Response
llm_response_str = """
{
  "thought_trace": "The topic is broad. I need to define 'Credit Risk Models' first.",
  "conviction_score": 0.92,
  "action": {
    "jsonrpc": "2.0",
    "method": "search_web",
    "params": {"query": "Quantum Computing Credit Risk Models"},
    "id": 1
  }
}
"""

# 5. Parse Response
output = plugin.parse_response(llm_response_str)

print(f"Conviction: {output.conviction_score}")
print(f"Action: {output.action}")
```

## Creating New Templates

Add new dictionary entries to `core/prompting/json_rpc_library.py` following the structure:

```python
NEW_TEMPLATE = {
    "name": "new_template_name",
    "description": "...",
    "template": "..."
}
```

Ensure the template instructs the model to output the `AdaptiveConvictionMetadata` structure (thought_trace, conviction_score, action).
