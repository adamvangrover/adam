# Orchestrator Agents Development Guide

This document provides guidelines and best practices for developing and maintaining Orchestrator Agents within the CreditSentry ecosystem.

## Role and Philosophy

The Orchestrator Agent is the "brain" and "central nervous system" of the entire agentic system. It is the highest level of control, responsible for interpreting user intent, formulating plans, delegating tasks to other agents, and synthesizing a final, coherent response.

**Core Principles:**

*   **Understand Intent, Don't Execute Tasks:** The primary role of the Orchestrator is to understand a user's goal and decompose it into a sequence of tasks. It should not perform the tasks itself.
*   **Master of Delegation:** The Orchestrator's main function is to know which Sub-Agent or Meta-Agent is the right tool for each task in its plan and to delegate accordingly.
*   **The Meta-Prompt is Law:** The Orchestrator's behavior is governed by its Meta-Prompt. This prompt is the constitution of the system. All delegation logic and operational constraints must be derived from it.
*   **Manage the Workflow:** The Orchestrator is responsible for managing the entire workflow, including handling dependencies between agent tasks (e.g., ensuring data is gathered before analysis begins) and managing errors or timeouts from individual agents.

## Modifying the Orchestrator

Modifying the Orchestrator is a high-risk activity that can have system-wide implications. Changes should be made with extreme care.

### The Meta-Prompt

*   **The Single Source of Truth:** The `META_PROMPT` class variable within the Orchestrator's code is the single source of truth for its behavior.
*   **Governance is Critical:** Any change to the `META_PROMPT` is a fundamental change to the system's governance and operational logic. All changes must be reviewed and approved by the "Meta-Prompt Governance Committee" as outlined in the main architectural documentation.
*   **Structure:** The Meta-Prompt is divided into clear components (`Core Directive`, `Agent Roster`, `Operational Constraints`, etc.). When making changes, adhere strictly to this structure.

### Delegation Logic

*   **LLM-Driven Planning:** The core delegation logic resides in the `execute` method. This method should use a powerful Large Language Model (LLM), guided by the `META_PROMPT`, to interpret the user's query and generate a step-by-step plan.
*   **Plan Representation:** The plan should be a structured object (e.g., a list of dictionaries), where each step defines the agent to be called, the inputs to provide, and any dependencies on previous steps.
*   **Agent Invocation:** The Orchestrator will then iterate through this plan, invoking the specified agents via their `execute` methods and passing the required data.

## Example: Credit Risk Orchestrator

This example shows how to create an Orchestrator Agent that assesses the credit risk of a company.

```python
from core.agents.agent_base import AgentBase

class CreditRiskOrchestrator(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.sub_agents = self.config.get("sub_agents", {})
        self.meta_agents = self.config.get("meta_agents", {})

    def execute(self, query):
        # 1. Decompose the query into a plan
        plan = self._create_plan(query)

        # 2. Execute the plan
        results = {}
        for step in plan:
            agent_name = step.get("agent")
            agent_input = results.get(step.get("input"))
            if agent_name in self.sub_agents:
                agent = self.sub_agents[agent_name]
                results[agent_name] = agent.execute(agent_input)
            elif agent_name in self.meta_agents:
                agent = self.meta_agents[agent_name]
                results[agent_name] = agent.execute(agent_input)

        # 3. Synthesize the results
        return self._synthesize_results(results)

    def _create_plan(self, query):
        # In a real implementation, this would use an LLM to generate the plan
        return [
            {
                "agent": "financial_news_sub_agent",
                "input": query,
            },
            {
                "agent": "sentiment_analysis_meta_agent",
                "input": "financial_news_sub_agent",
            },
        ]

    def _synthesize_results(self, results):
        # In a real implementation, this would use an LLM to synthesize the results
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.9,
            "data": results,
        }

```

### Configuration

The `CreditRiskOrchestrator` would be configured in the `config/agents.yaml` file as follows:

```yaml
credit_sentry_agents:
  orchestrators:
    credit_risk_orchestrator:
      persona: "An Orchestrator Agent that assesses the credit risk of a company."
      description: "This agent is responsible for assessing the credit risk of a company."
      expertise: "Credit risk assessment"
      config:
        sub_agents:
          financial_news_sub_agent: ...
        meta_agents:
          sentiment_analysis_meta_agent: ...
```

## Best Practices

*   **Statelessness:** The Orchestrator should ideally be stateless within a single user request. The state of the workflow is managed by the plan it generates and the results it collects.
*   **Error Handling:** The Orchestrator must have robust error handling. If an agent fails, the Orchestrator should decide whether to retry, abort the plan, or continue with a partial result, logging the event clearly.
*   **Testing:** Testing the Orchestrator is complex. It requires end-to-end tests that simulate a user query and verify that the correct sequence of agents is called and that the final output is synthesized correctly.
