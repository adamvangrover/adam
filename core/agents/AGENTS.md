**File name:** `AGENTS.md`

**File path:** `core/agents/AGENTS.md`

```markdown
# Adam v17.0 Agents Overview

This document provides an overview of the various agents that comprise the Adam v17.0 system. Each agent is a specialized AI module responsible for a specific aspect of financial analysis, risk assessment, or knowledge management.

## Agent Categories

Adam v17.0's agents can be broadly categorized as follows:

* **Market Analysis Agents:** Analyze market trends, sentiment, and events.
    * Examples: `market_sentiment_agent`, `macroeconomic_analysis_agent`, `geopolitical_risk_agent`
* **Fundamental Analysis Agents:** Perform in-depth analysis of company financials and fundamentals.
    * Examples: `fundamental_analyst_agent`
* **Technical Analysis Agents:** Analyze price charts, technical indicators, and patterns.
    * Examples: `technical_analyst_agent`
* **Risk Management Agents:** Assess and manage various types of investment risks.
    * Examples: `risk_assessment_agent`
* **Knowledge Management Agents:** Manage, update, and refine the knowledge base.
    * Examples: `lexica_agent`, `archive_manager_agent`
* **System Enhancement Agents:** Improve the overall performance and capabilities of the Adam v17.0 system.
    * Examples: `agent_forge`, `prompt_tuner`, `code_alchemist`
* **Communication and Interface Agents:** Facilitate communication and interaction with users.
    * Examples: `newsletter_layout_specialist_agent`, `echo_agent`

## Agent Configuration

Each agent is configured through the `config/agents.yaml` file. The configuration includes:

* **Persona:** A descriptive name that reflects the agent's role and personality.
* **Description:** A brief overview of the agent's purpose and capabilities.
* **Expertise:** A list of the agent's areas of expertise.
* **Data Sources:** The data sources that the agent uses for its analysis.
* **Alerting Thresholds:** Thresholds for triggering alerts based on specific events or conditions.
* **Communication Style:** The agent's preferred communication style (e.g., concise, detailed, visual).
* **Knowledge Graph Integration:** Whether the agent integrates with the knowledge graph.
* **API Integration:** Whether the agent interacts with the Adam v17.0 API.
* **Other Agent-Specific Configurations:** Additional parameters and settings specific to the agent's functionality.

## Agent Interaction and Collaboration

Agents interact and collaborate with each other through various mechanisms, including:

* **Message Queue:** Agents can send and receive messages through a message queue to share information and coordinate tasks.
* **API:** Agents can interact with the Adam v17.0 API to access data, run analysis modules, and generate reports.
* **Knowledge Graph:** Agents can access and update the knowledge graph to share knowledge and enhance their analysis.

## Agent Development and Refinement

The Adam v17.0 system is designed to be continuously improved and refined through:

* **Agent Forge:** The `agent_forge` agent can create new agents or modify existing ones based on user needs and feedback.
* **Prompt Tuner:** The `prompt_tuner` agent can refine agent prompts and communication styles to improve clarity and efficiency.
* **Code Alchemist:** The `code_alchemist` agent can optimize agent code for performance and scalability.
* **User Feedback:** Users can provide feedback on agent performance and suggest improvements.

## Example Agent Configurations

Here are a few examples of agent configurations from the `config/agents.yaml` file:

```yaml
# Market Sentiment Agent
market_sentiment_agent:
  #... configuration details...

# Fundamental Analyst Agent
fundamental_analyst_agent:
  #... configuration details...

# Risk Assessment Agent
risk_assessment_agent:
  #... configuration details...
```

## Detailed Outlines for Complex Agents

For more complex agents, such as the `agent_forge` or `echo_agent`, detailed outlines and explanations of their functionalities and configurations can be found in separate documentation files.

This overview provides a general understanding of Adam v17.0's agents and their roles within the system. For more specific information and guidance on individual agents, refer to the `config/agents.yaml` file and other relevant documentation.
```
