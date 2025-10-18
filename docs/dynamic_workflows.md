# Dynamic Workflow Generation

## Overview

Adam v22.0 can dynamically generate novel workflows to answer complex user queries that are not covered by predefined workflows. This is achieved using the `WorkflowCompositionSkill`, a Semantic Kernel skill that allows the Agent Orchestrator to reason about the available agent skills and compose them into a coherent workflow.

## How it Works

1.  If no predefined workflow matches the user's query, the orchestrator invokes the `WorkflowCompositionSkill`.
2.  The `WorkflowCompositionSkill` takes the user's query and the list of all available agent skills as input.
3.  The skill's prompt instructs the LLM to generate a workflow in the same YAML format as the predefined workflows.
4.  The skill includes functions for validating the generated workflow to ensure it is syntactically correct and logically sound.
5.  The orchestrator then executes this dynamically generated workflow.

## Example

**User query:** "What is the current sentiment of the market towards Apple stock?"

**Dynamically generated workflow:**

```yaml
agents:
  - MarketSentimentAgent
dependencies: {}
```
