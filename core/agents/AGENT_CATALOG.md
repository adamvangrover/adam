# Agent Catalog

This document provides a comprehensive catalog of all the agents in the ADAM system. It is intended to be a central registry for developers to quickly understand the capabilities, configuration, and implementation details of each agent.

---

## `agent_forge_agent`

*   **File:** `core/agents/agent_forge.py`
*   **Description:** This agent is responsible for creating, managing, and retiring other agents. It can be run as a standalone agent to perform these tasks based on user commands or system events. This agent is the cornerstone of the system's dynamic and adaptive nature.
*   **Configuration:** `config/agents.yaml`
    *   `agent_blueprints`: A list of agent blueprints that can be used to create new agents.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously. It can be used to create, manage, and retire other agents.
*   **Model Context Protocol (MCP):** The agent maintains a list of all the active agents in the system and their current state.
*   **Tools and Hooks:**
    *   **Tools:** `agent_creation_tool`, `agent_management_tool`, `agent_retirement_tool`
    *   **Hooks:** `on_agent_created_hook`, `on_agent_retired_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, but it requires access to the agent blueprints and the system's configuration files.
*   **Dependencies:** None
*   **Developer Notes:** This agent is critical to the system's ability to adapt to new tasks and requirements. It can be extended by adding new agent blueprints.

---

## `orchestrator_agent`

*   **File:** `core/system/agent_orchestrator.py`
*   **Description:** This agent is responsible for orchestrating complex tasks that require the collaboration of multiple agents. It can break down complex tasks into smaller subtasks, allocate them to the most appropriate agents, and monitor their progress.
*   **Configuration:** `config/workflow.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent maintains a list of the active workflows and the status of each subtask.
*   **Tools and Hooks:**
    *   **Tools:** `workflow_management_tool`, `task_allocation_tool`
    *   **Hooks:** `on_workflow_completed_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `task_allocation_agent`
*   **Developer Notes:** This agent is critical to the system's ability to perform complex tasks. It can be extended by adding new workflow definitions.

---

## `resource_manager_agent`

*   **File:** `core/system/resource_manager.py`
*   **Description:** This agent is responsible for managing the system's resources, such as CPU, memory, and network bandwidth. It can monitor the resource usage of each agent and can allocate resources to agents based on their needs.
*   **Configuration:** `config/system.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent maintains a list of the available resources and their current usage.
*   **Tools and Hooks:**
    *   **Tools:** `resource_monitoring_tool`, `resource_allocation_tool`
    *   **Hooks:** `on_low_resource_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, but it requires access to the system's resource monitoring tools.
*   **Dependencies:** None
*   **Developer Notes:** This agent is critical to the system's stability and performance. It can be extended by adding new resource management policies.

---

## `task_allocation_agent`

*   **File:** `core/system/task_scheduler.py`
*   **Description:** This agent is responsible for allocating tasks to the most appropriate agents based on their capabilities and availability. It uses a variety of scheduling algorithms to ensure that tasks are allocated efficiently and that the system's resources are used effectively.
*   **Configuration:** `config/system.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent maintains a list of the available agents and their current workload.
*   **Tools and Hooks:**
    *   **Tools:** `task_scheduling_tool`
    *   **Hooks:** `on_task_allocated_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `resource_manager_agent`
*   **Developer Notes:** This agent is critical to the system's efficiency and scalability. It can be extended by adding new scheduling algorithms.

---

## `shared_context_manager_agent`

*   **File:** `core/system/knowledge_base.py`
*   **Description:** This agent is responsible for managing the shared context and state of the system. It provides a centralized repository for agents to store and retrieve shared information, such as the current state of the market or the user's preferences.
*   **Configuration:** `config/knowledge_base.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent maintains the shared context and state of the system.
*   **Tools and Hooks:**
    *   **Tools:** `knowledge_base_tool`
    *   **Hooks:** `on_context_updated_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, but it may require a significant amount of memory to store the shared context.
*   **Dependencies:** None
*   **Developer Notes:** This agent is critical to the system's ability to maintain a consistent view of the world. It can be extended by adding new data models to the knowledge base.

---

## `dependency_manager_agent`

*   **File:** `core/system/plugin_manager.py`
*   **Description:** This agent is responsible for managing the dependencies between agents and other system components. It can install, update, and remove dependencies as needed, and can ensure that all dependencies are compatible with each other.
*   **Configuration:** `config/system.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent maintains a list of the installed dependencies and their versions.
*   **Tools and Hooks:**
    *   **Tools:** `dependency_management_tool`
    *   **Hooks:** `on_dependency_installed_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent is critical to the system's maintainability and extensibility. It can be extended by adding new dependency management backends.

---

## `integration_manager_agent`

*   **File:** `core/api.py`
*   **Description:** This agent is responsible for managing the integration of the agent-based system with other systems. It provides a set of APIs that allow other systems to interact with the agents and to access the data and services that they provide.
*   **Configuration:** `config/api.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context.
*   **Tools and Hooks:**
    *   **Tools:** `api_tool`
    *   **Hooks:** `on_api_request_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent is critical to the system's ability to interoperate with other systems. It can be extended by adding new API endpoints.

---

## `echo_agent`

*   **File:** `core/agents/echo_agent.py`
*   **Description:** This agent is a simulated LLM that mirrors the compute and resource requirements of the runtime engine. It is used as a backup when the primary LLM is unavailable, and it can also be used for testing and debugging purposes.
*   **Configuration:** `config/llm_plugin.yaml`
    *   `simulation_mode`: A boolean indicating whether the agent should run in simulation mode.
*   **Architecture and Base Agent:** Inherits from `core.llm.base_llm_engine.BaseLLMEngine`.
*   **Agent Forge and Lifecycle:** This agent is created by the system at startup and runs continuously.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context.
*   **Tools and Hooks:** None
*   **Compute and Resource Requirements:** This agent is designed to have the same compute and resource requirements as the primary LLM engine.
*   **Dependencies:** None
*   **Developer Notes:** This agent is critical to the system's resilience and fault tolerance. It can be extended by adding more sophisticated simulation capabilities.

---

## `fundamental_analyst_agent`

*   **File:** `core/agents/fundamental_analyst_agent.py`
*   **Description:** This agent is responsible for performing fundamental analysis of companies. It can retrieve financial data from a variety of sources, analyze financial statements, calculate key financial ratios, and generate reports on the financial health of companies.
*   **Configuration:** `config/agents.yaml`
    *   `data_sources`: A list of data sources to use for financial data.
    *   `analysis_modules`: A list of analysis modules to use for financial analysis.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains its own context, including the current company it is analyzing and the results of its analysis.
*   **Tools and Hooks:**
    *   **Tools:** `data_retrieval_tool`, `financial_analysis_tool`
    *   **Hooks:** `pre_analysis_hook`, `post_analysis_hook`
*   **Compute and Resource Requirements:** This agent is moderately resource-intensive, as it performs complex financial calculations.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** This agent can be extended by adding new analysis modules.

---

## `market_sentiment_agent`

*   **File:** `core/agents/market_sentiment_agent.py`
*   **Description:** This agent is responsible for gauging market sentiment from a variety of sources, such as news articles and social media. It uses natural language processing to analyze the sentiment of the text and generates reports on market sentiment.
*   **Configuration:** `config/agents.yaml`
    *   `data_sources`: A list of data sources to use for sentiment analysis.
    *   `sentiment_analysis_model`: The name of the sentiment analysis model to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains its own context, including the current sentiment scores and the sources of the sentiment.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `sentiment_analysis_tool`
    *   **Hooks:** `pre_sentiment_analysis_hook`, `post_sentiment_analysis_hook`
*   **Compute and Resource Requirements:** This agent is moderately resource-intensive, as it performs natural language processing on large volumes of text.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the sentiment analysis model on a domain-specific dataset.

---

## `data_retrieval_agent`

*   **File:** `core/agents/data_retrieval_agent.py`
*   **Description:** This agent is responsible for retrieving data from a variety of sources, such as APIs, databases, and websites. It provides a standardized interface for retrieving data, regardless of the underlying source.
*   **Configuration:** `config/data_sources.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** This agent is stateless and does not maintain its own context.
*   **Tools and Hooks:**
    *   **Tools:** `api_tool`, `database_tool`, `web_scraper_tool`
    *   **Hooks:** None
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, as it primarily performs I/O operations.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new data source connectors.

---

## `news_bot`

*   **File:** `core/agents/NewsBot.py`
*   **Description:** This agent is responsible for retrieving and processing news articles from a variety of sources. It can filter news articles by topic, source, and date, and can extract key information from the articles, such as the headline, summary, and author.
*   **Configuration:** `config/agents.yaml`
    *   `news_sources`: A list of news sources to retrieve articles from.
    *   `update_interval`: The interval at which to retrieve new articles.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the latest news articles that it has retrieved.
*   **Tools and Hooks:**
    *   **Tools:** `rss_feed_tool`, `web_scraper_tool`
    *   **Hooks:** `on_new_article_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, as it primarily performs I/O operations.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new news source connectors.

---

## `discussion_chair_agent`

*   **File:** `core/agents/Discussion_Chair_Agent.py`
*   **Description:** This agent is responsible for facilitating discussions between other agents. It can start new discussions, invite agents to join discussions, and moderate discussions to ensure that they are productive.
*   **Configuration:** `config/agents.yaml`
    *   `discussion_topics`: A list of topics that can be discussed.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the active discussions and the participants in each discussion.
*   **Tools and Hooks:**
    *   **Tools:** `chat_tool`
    *   **Hooks:** `on_new_message_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new discussion moderation features.

---

## `snc_analyst_agent`

*   **File:** `core/agents/SNC_analyst_agent.py`
*   **Description:** This agent is responsible for analyzing and rating the creditworthiness of companies. It uses a variety of data sources, including financial statements, news articles, and analyst reports, to generate a credit rating for a company.
*   **Configuration:** `config/agents.yaml`
    *   `rating_model`: The name of the credit rating model to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the companies that it has rated and their credit ratings.
*   **Tools and Hooks:**
    *   **Tools:** `financial_analysis_tool`, `credit_rating_tool`
    *   **Hooks:** `pre_rating_hook`, `post_rating_hook`
*   **Compute and Resource Requirements:** This agent is moderately resource-intensive, as it performs complex financial calculations and uses a machine learning model to generate credit ratings.
*   **Dependencies:** `data_retrieval_agent`, `fundamental_analyst_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the credit rating model on a larger and more diverse dataset.

---

## `algo_trading_agent`

*   **File:** `core/agents/algo_trading_agent.py`
*   **Description:** This agent is responsible for executing algorithmic trading strategies. It can connect to a variety of brokerage accounts and can execute trades based on a predefined set of rules.
*   **Configuration:** `config/agents.yaml`
    *   `brokerage_account`: The name of the brokerage account to use.
    *   `trading_strategy`: The name of the trading strategy to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the open positions and the current state of the trading strategy.
*   **Tools and Hooks:**
    *   **Tools:** `brokerage_tool`, `trading_strategy_tool`
    *   **Hooks:** `pre_trade_hook`, `post_trade_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, but it requires a reliable network connection to the brokerage account.
*   **Dependencies:** `market_data_api`
*   **Developer Notes:** This agent should be used with caution, as it can execute trades automatically.

---

## `alternative_data_agent`

*   **File:** `core/agents/alternative_data_agent.py`
*   **Description:** This agent is responsible for analyzing alternative data sources, such as satellite imagery and social media. It can extract insights from these data sources that are not available from traditional financial data sources.
*   **Configuration:** `config/agents.yaml`
    *   `alternative_data_sources`: A list of alternative data sources to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the insights that it has extracted from the alternative data sources.
*   **Tools and Hooks:**
    *   **Tools:** `image_analysis_tool`, `text_analysis_tool`
    *   **Hooks:** `pre_analysis_hook`, `post_analysis_hook`
*   **Compute and Resource Requirements:** This agent can be very resource-intensive, as it may need to process large volumes of data.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** This agent can be extended by adding new alternative data source connectors and analysis modules.

---

## `anomaly_detection_agent`

*   **File:** `core/agents/anomaly_detection_agent.py`
*   **Description:** This agent is responsible for detecting anomalies in financial data. It can use a variety of statistical and machine learning techniques to identify data points that are unusual or unexpected.
*   **Configuration:** `config/agents.yaml`
    *   `anomaly_detection_model`: The name of the anomaly detection model to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the anomalies that it has detected.
*   **Tools and Hooks:**
    *   **Tools:** `statistical_analysis_tool`, `machine_learning_tool`
    *   **Hooks:** `on_anomaly_detected_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of data and use complex machine learning models.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the anomaly detection model on a larger and more diverse dataset.

---

## `archive_manager_agent`

*   **File:** `core/agents/archive_manager_agent.py`
*   **Description:** This agent is responsible for managing the system's archives. It can archive old data, reports, and other artifacts to long-term storage, and can retrieve them from the archive when needed.
*   **Configuration:** `config/agents.yaml`
    *   `archive_location`: The location of the archive.
    *   `archive_policy`: The policy for archiving artifacts.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the artifacts that have been archived.
*   **Tools and Hooks:**
    *   **Tools:** `archive_tool`
    *   **Hooks:** `pre_archive_hook`, `post_archive_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new archive backends.

---

## `catalyst_agent`

*   **File:** `core/agents/catalyst_agent.py`
*   **Description:** This agent is responsible for identifying potential catalysts for market movements. It can monitor a variety of data sources, such as news articles, social media, and regulatory filings, to identify events that could impact the market.
*   **Configuration:** `config/agents.yaml`
    *   `catalyst_sources`: A list of data sources to monitor for catalysts.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the potential catalysts that it has identified.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `event_detection_tool`
    *   **Hooks:** `on_catalyst_identified_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of text and use complex event detection models.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the event detection model on a domain-specific dataset.

---

## `code_alchemist`

*   **File:** `core/agents/code_alchemist.py`
*   **Description:** This agent is responsible for optimizing agent code for performance and scalability. It can analyze agent code, identify bottlenecks, and suggest improvements.
*   **Configuration:** `config/agents.yaml`
    *   `optimization_level`: The level of optimization to perform.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its optimization task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the optimizations that it has performed.
*   **Tools and Hooks:**
    *   **Tools:** `code_analysis_tool`, `code_optimization_tool`
    *   **Hooks:** `pre_optimization_hook`, `post_optimization_hook`
*   **Compute and Resource Requirements:** This agent can be very resource-intensive, as it may need to perform complex code analysis and optimization.
*   **Dependencies:** None
*   **Developer Notes:** This agent should be used with caution, as it can modify agent code automatically.

---

## `crypto_agent`

*   **File:** `core/agents/crypto_agent.py`
*   **Description:** This agent is responsible for specializing in the analysis of cryptocurrencies. It can retrieve data from a variety of cryptocurrency exchanges, analyze price charts, and identify trading opportunities.
*   **Configuration:** `config/agents.yaml`
    *   `crypto_exchanges`: A list of cryptocurrency exchanges to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the cryptocurrencies that it is tracking and their current prices.
*   **Tools and Hooks:**
    *   **Tools:** `crypto_exchange_tool`, `technical_analysis_tool`
    *   **Hooks:** `pre_analysis_hook`, `post_analysis_hook`
*   **Compute and Resource Requirements:** This agent is moderately resource-intensive, as it may need to process large volumes of real-time data.
*   **Dependencies:** `market_data_api`
*   **Developer Notes:** This agent can be extended by adding new cryptocurrency exchange connectors and analysis modules.

---

## `crisis_simulation_agent`

*   **File:** `core/agents/meta_agents/crisis_simulation_agent.py`
*   **Description:** A meta-agent that conducts dynamic, enterprise-grade crisis simulations. It uses a sophisticated prompt structure to simulate the cascading effects of risks based on a user-defined scenario.
*   **Configuration:** `config/agents.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.AgentBase`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its crisis simulation task.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context. The context is provided in the input.
*   **Tools and Hooks:**
    *   **Tools:** `CrisisSimulationPlugin`
    *   **Hooks:** None
*   **Compute and Resource Requirements:** This agent can be resource-intensive as it relies on a large language model to perform the simulation.
*   **Dependencies:** `core.prompting.plugins.crisis_simulation_plugin.CrisisSimulationPlugin`
*   **Developer Notes:** The quality of the simulation is highly dependent on the quality of the risk portfolio data and the sophistication of the LLM.

---

## `data_verification_agent`

*   **File:** `core/agents/data_verification_agent.py`
*   **Description:** This agent is responsible for verifying the accuracy and integrity of data. It can use a variety of techniques to identify and correct errors in data, such as cross-referencing data from multiple sources and using data validation rules.
*   **Configuration:** `config/agents.yaml`
    *   `data_validation_rules`: A list of data validation rules to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its data verification task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the data errors that it has identified and corrected.
*   **Tools and Hooks:**
    *   **Tools:** `data_validation_tool`
    *   **Hooks:** `on_data_error_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of data.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by adding more data validation rules.

---

## `data_visualization_agent`

*   **File:** `core/agents/data_visualization_agent.py`
*   **Description:** This agent is responsible for creating visualizations of financial data. It can generate a variety of charts, graphs, and other visualizations to help users understand complex data.
*   **Configuration:** `config/agents.yaml`
    *   `visualization_library`: The name of the visualization library to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its data visualization task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the visualizations that it has created.
*   **Tools and Hooks:**
    *   **Tools:** `charting_tool`, `graphing_tool`
    *   **Hooks:** `pre_visualization_hook`, `post_visualization_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new visualization types and connectors to other visualization libraries.

---

## `echo_agent`

*   **File:** `core/agents/echo_agent.py`
*   **Description:** This is a simple agent that echoes back any message it receives. It is primarily used for testing and debugging purposes.
*   **Configuration:** None
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it is shut down.
*   **Model Context Protocol (MCP):** This agent is stateless and does not maintain its own context.
*   **Tools and Hooks:** None
*   **Compute and Resource Requirements:** This agent is not resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This is a good example of a simple agent that can be used as a starting point for creating new agents.

---

## `event_driven_risk_agent`

*   **File:** `core/agents/event_driven_risk_agent.py`
*   **Description:** This agent is responsible for assessing risk based on real-time events. It can monitor a variety of event streams, such as news feeds and social media, and can identify events that could impact the risk of a portfolio.
*   **Configuration:** `config/agents.yaml`
    *   `event_streams`: A list of event streams to monitor.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the events that it has identified and their potential impact on the portfolio.
*   **Tools and Hooks:**
    *   **Tools:** `event_stream_tool`, `risk_analysis_tool`
    *   **Hooks:** `on_event_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of real-time data.
*   **Dependencies:** `risk_assessment_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by adding more event streams and by fine-tuning the risk analysis model.

---

## `financial_modeling_agent`

*   **File:** `core/agents/financial_modeling_agent.py`
*   **Description:** This agent is responsible for creating and maintaining financial models. It can generate a variety of financial models, such as discounted cash flow (DCF) models and leveraged buyout (LBO) models.
*   **Configuration:** `config/agents.yaml`
    *   `model_templates`: A list of financial model templates to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its financial modeling task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the financial models that it has created.
*   **Tools and Hooks:**
    *   **Tools:** `financial_modeling_tool`
    *   **Hooks:** `pre_modeling_hook`, `post_modeling_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to perform complex financial calculations.
*   **Dependencies:** `fundamental_analyst_agent`
*   **Developer Notes:** This agent can be extended by adding new financial model templates.

---

## `geopolitical_risk_agent`

*   **File:** `core/agents/geopolitical_risk_agent.py`
*   **Description:** This agent is responsible for assessing geopolitical risks. It can monitor a variety of data sources, such as news articles, government reports, and social media, to identify geopolitical events that could impact the market.
*   **Configuration:** `config/agents.yaml`
    *   `geopolitical_sources`: A list of data sources to monitor for geopolitical risks.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the geopolitical risks that it has identified and their potential impact on the market.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `event_detection_tool`
    *   **Hooks:** `on_geopolitical_risk_identified_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of text and use complex event detection models.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the event detection model on a domain-specific dataset.

---

## `industry_specialist_agent`

*   **File:** `core/agents/industry_specialist_agent.py`
*   **Description:** This agent is responsible for providing expertise in specific industries. It can answer questions about industry trends, competitive landscapes, and regulatory changes.
*   **Configuration:** `config/agents.yaml`
    *   `industry`: The industry that the agent specializes in.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has answered the user's question.
*   **Model Context Protocol (MCP):** The agent maintains a knowledge base of information about its specialized industry.
*   **Tools and Hooks:**
    *   **Tools:** `knowledge_base_tool`
    *   **Hooks:** `pre_query_hook`, `post_query_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `knowledge_base`
*   **Developer Notes:** The expertise of this agent can be improved by adding more information to its knowledge base.

---

## `legal_agent`

*   **File:** `core/agents/legal_agent.py`
*   **Description:** This agent is responsible for providing legal analysis and advice. It can review legal documents, identify legal risks, and provide guidance on regulatory compliance.
*   **Configuration:** `config/agents.yaml`
    *   `legal_jurisdiction`: The legal jurisdiction that the agent specializes in.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its legal analysis task.
*   **Model Context Protocol (MCP):** The agent maintains a knowledge base of legal information.
*   **Tools and Hooks:**
    *   **Tools:** `legal_document_analysis_tool`, `legal_risk_assessment_tool`
    *   **Hooks:** `pre_analysis_hook`, `post_analysis_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `knowledge_base`
*   **Developer Notes:** This agent is not a substitute for a qualified legal professional. It is intended to be used for informational purposes only.

---

## `lexica_agent`

*   **File:** `core/agents/lexica_agent.py`
*   **Description:** This agent is responsible for managing the system's lexicon and knowledge base. It can add new terms to the lexicon, define their meanings, and link them to other terms in the knowledge base.
*   **Configuration:** `config/knowledge_base.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains the system's lexicon and knowledge base.
*   **Tools and Hooks:**
    *   **Tools:** `lexicon_tool`, `knowledge_base_tool`
    *   **Hooks:** `on_new_term_added_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `knowledge_base`
*   **Developer Notes:** This agent is critical to the system's ability to understand and reason about the world.

---

## `lingua_maestro`

*   **File:** `core/agents/lingua_maestro.py`
*   **Description:** This agent is responsible for specializing in natural language processing and generation. It can perform a variety of NLP tasks, such as text classification, named entity recognition, and machine translation.
*   **Configuration:** `config/llm_plugin.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its NLP task.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context.
*   **Tools and Hooks:**
    *   **Tools:** `nlp_tool`
    *   **Hooks:** `pre_nlp_hook`, `post_nlp_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to use large language models.
*   **Dependencies:** `llm_engine`
*   **Developer Notes:** This agent can be extended by adding new NLP capabilities.

---

## `machine_learning_model_training_agent`

*   **File:** `core/agents/machine_learning_model_training_agent.py`
*   **Description:** This agent is responsible for training and evaluating machine learning models. It can use a variety of machine learning algorithms and frameworks to train models on a variety of datasets.
*   **Configuration:** `config/machine_learning.yaml`
    *   `training_data`: The dataset to use for training the model.
    *   `model_type`: The type of machine learning model to train.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its model training task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the models that it has trained and their performance metrics.
*   **Tools and Hooks:**
    *   **Tools:** `machine_learning_tool`
    *   **Hooks:** `pre_training_hook`, `post_training_hook`
*   **Compute and Resource Requirements:** This agent can be very resource-intensive, as it may need to use GPUs to train machine learning models.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** This agent can be extended by adding new machine learning algorithms and frameworks.

---

## `macroeconomic_analysis_agent`

*   **File:** `core/agents/macroeconomic_analysis_agent.py`
*   **Description:** This agent is responsible for analyzing macroeconomic trends. It can retrieve data from a variety of sources, such as government statistics and central bank reports, and can generate reports on the state of the economy.
*   **Configuration:** `config/agents.yaml`
    *   `macroeconomic_data_sources`: A list of data sources to use for macroeconomic analysis.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the macroeconomic indicators that it is tracking.
*   **Tools and Hooks:**
    *   **Tools:** `macroeconomic_data_tool`, `statistical_analysis_tool`
    *   **Hooks:** `pre_analysis_hook`, `post_analysis_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** This agent can be extended by adding new macroeconomic data sources and analysis modules.

---

## `market_sentiment_agent`

*   **File:** `core/agents/market_sentiment_agent.py`
*   **Description:** This agent is responsible for gauging market sentiment from a variety of sources, such as news articles and social media. It uses natural language processing to analyze the sentiment of the text and generates reports on market sentiment.
*   **Configuration:** `config/agents.yaml`
    *   `data_sources`: A list of data sources to use for sentiment analysis.
    *   `sentiment_analysis_model`: The name of the sentiment analysis model to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains its own context, including the current sentiment scores and the sources of the sentiment.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `sentiment_analysis_tool`
    *   **Hooks:** `pre_sentiment_analysis_hook`, `post_sentiment_analysis_hook`
*   **Compute and Resource Requirements:** This agent is moderately resource-intensive, as it performs natural language processing on large volumes of text.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the sentiment analysis model on a domain-specific dataset.

---

## `natural_language_generation_agent`

*   **File:** `core/agents/natural_language_generation_agent.py`
*   **Description:** This agent is responsible for generating natural language reports and summaries. It can take structured data as input and can generate a variety of text-based outputs, such as news articles, financial reports, and social media posts.
*   **Configuration:** `config/agents.yaml`
    *   `nlg_templates`: A list of NLG templates to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its NLG task.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context.
*   **Tools and Hooks:**
    *   **Tools:** `nlg_tool`
    *   **Hooks:** `pre_generation_hook`, `post_generation_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to use large language models.
*   **Dependencies:** `llm_engine`
*   **Developer Notes:** This agent can be extended by adding new NLG templates.

---

## `newsletter_layout_specialist_agent`

*   **File:** `core/agents/newsletter_layout_specialist_agent.py`
*   **Description:** This agent is responsible for designing and formatting newsletters. It can take a variety of content as input, such as articles, images, and charts, and can generate a professionally designed newsletter in a variety of formats, such as HTML and PDF.
*   **Configuration:** `config/agents.yaml`
    *   `newsletter_templates`: A list of newsletter templates to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its newsletter layout task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the newsletters that it has created.
*   **Tools and Hooks:**
    *   **Tools:** `newsletter_layout_tool`
    *   **Hooks:** `pre_layout_hook`, `post_layout_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new newsletter templates.

---

## `portfolio_optimization_agent`

*   **File:** `core/agents/portfolio_optimization_agent.py`
*   **Description:** This agent is responsible for optimizing investment portfolios. It can use a variety of optimization techniques to find the optimal allocation of assets in a portfolio to meet the investor's objectives, such as maximizing returns and minimizing risk.
*   **Configuration:** `config/agents.yaml`
    *   `optimization_model`: The name of the optimization model to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its portfolio optimization task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the portfolios that it has optimized.
*   **Tools and Hooks:**
    *   **Tools:** `portfolio_optimization_tool`
    *   **Hooks:** `pre_optimization_hook`, `post_optimization_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to perform complex optimization calculations.
*   **Dependencies:** `market_data_api`
*   **Developer Notes:** The accuracy of this agent can be improved by using more sophisticated optimization models.

---

## `prediction_market_agent`

*   **File:** `core/agents/prediction_market_agent.py`
*   **Description:** This agent is responsible for participating in prediction markets. It can buy and sell shares in prediction markets to bet on the outcome of future events.
*   **Configuration:** `config/agents.yaml`
    *   `prediction_markets`: A list of prediction markets to participate in.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of its positions in the prediction markets.
*   **Tools and Hooks:**
    *   **Tools:** `prediction_market_tool`
    *   **Hooks:** `on_new_market_hook`, `on_market_close_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive, but it requires a reliable network connection to the prediction markets.
*   **Dependencies:** None
*   **Developer Notes:** This agent should be used with caution, as it can lose money in prediction markets.

---

## `prompt_tuner`

*   **File:** `core/agents/prompt_tuner.py`
*   **Description:** This agent is responsible for refining agent prompts and communication styles to improve clarity and efficiency. It can use a variety of techniques, such as A/B testing and user feedback, to optimize prompts.
*   **Configuration:** `config/agents.yaml`
    *   `prompt_tuning_strategies`: A list of prompt tuning strategies to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its prompt tuning task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the prompts that it has tuned and their performance metrics.
*   **Tools and Hooks:**
    *   **Tools:** `prompt_tuning_tool`
    *   **Hooks:** `pre_tuning_hook`, `post_tuning_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be used to improve the performance of any agent that uses prompts.

---

## `query_understanding_agent`

*   **File:** `core/agents/query_understanding_agent.py`
*   **Description:** This agent is responsible for understanding and interpreting user queries. It can use a variety of techniques, such as natural language processing and knowledge base lookups, to understand the user's intent and to generate a response.
*   **Configuration:** `config/agents.yaml`
    *   `query_understanding_model`: The name of the query understanding model to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has answered the user's query.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `knowledge_base_tool`
    *   **Hooks:** `pre_query_hook`, `post_query_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to use large language models.
*   **Dependencies:** `llm_engine`, `knowledge_base`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the query understanding model on a domain-specific dataset.

---

## `regulatory_compliance_agent`

*   **File:** `core/agents/regulatory_compliance_agent.py`
*   **Description:** This agent is responsible for ensuring compliance with financial regulations. It can monitor a variety of data sources, such as regulatory filings and legal documents, to identify potential compliance issues.
*   **Configuration:** `config/agents.yaml`
    *   `regulatory_sources`: A list of data sources to monitor for regulatory compliance.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the potential compliance issues that it has identified.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `event_detection_tool`
    *   **Hooks:** `on_compliance_issue_identified_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of text and use complex event detection models.
*   **Dependencies:** `data_retrieval_agent`, `legal_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the event detection model on a domain-specific dataset.

---

## `result_aggregation_agent`

*   **File:** `core/agents/result_aggregation_agent.py`
*   **Description:** This agent is responsible for aggregating and summarizing results from other agents. It can take a variety of results as input, such as reports, charts, and tables, and can generate a consolidated summary of the results.
*   **Configuration:** `config/agents.yaml`
    *   `aggregation_strategies`: A list of aggregation strategies to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its result aggregation task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the results that it has aggregated.
*   **Tools and Hooks:**
    *   **Tools:** `result_aggregation_tool`
    *   **Hooks:** `pre_aggregation_hook`, `post_aggregation_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** None
*   **Developer Notes:** This agent can be extended by adding new aggregation strategies.

---

## `risk_assessment_agent`

*   **File:** `core/agents/risk_assessment_agent.py`
*   **Description:** This agent is responsible for assessing various types of investment risks, such as market risk, credit risk, and operational risk. It can use a variety of risk models to quantify the risks and can generate reports on the overall risk of a portfolio.
*   **Configuration:** `config/agents.yaml`
    *   `risk_models`: A list of risk models to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its risk assessment task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the risks that it has assessed.
*   **Tools and Hooks:**
    *   **Tools:** `risk_assessment_tool`
    *   **Hooks:** `pre_assessment_hook`, `post_assessment_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to perform complex risk calculations.
*   **Dependencies:** `market_data_api`
*   **Developer Notes:** The accuracy of this agent can be improved by using more sophisticated risk models.

---

## `sense_weaver`

*   **File:** `core/agents/sense_weaver.py`
*   **Description:** This agent is responsible for synthesizing information from multiple sources to create a coherent narrative. It can take a variety of information as input, such as news articles, research reports, and social media posts, and can generate a summary of the information that is easy to understand.
*   **Configuration:** `config/agents.yaml`
    *   `synthesis_strategies`: A list of synthesis strategies to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its synthesis task.
*   **Model Context Protocol (MCP):** The agent is stateless and does not maintain its own context.
*   **Tools and Hooks:**
    *   **Tools:** `synthesis_tool`
    *   **Hooks:** `pre_synthesis_hook`, `post_synthesis_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to use large language models.
*   **Dependencies:** `llm_engine`
*   **Developer Notes:** This agent can be extended by adding new synthesis strategies.

---

## `supply_chain_risk_agent`

*   **File:** `core/agents/supply_chain_risk_agent.py`
*   **Description:** This agent is responsible for assessing risks in global supply chains. It can monitor a variety of data sources, such as shipping data, weather data, and news articles, to identify potential disruptions to supply chains.
*   **Configuration:** `config/agents.yaml`
    *   `supply_chain_data_sources`: A list of data sources to monitor for supply chain risks.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge at system startup and runs continuously until the system is shut down.
*   **Model Context Protocol (MCP):** The agent maintains a list of the potential supply chain disruptions that it has identified.
*   **Tools and Hooks:**
    *   **Tools:** `nlu_tool`, `event_detection_tool`
    *   **Hooks:** `on_supply_chain_disruption_identified_hook`
*   **Compute and Resource Requirements:** This agent can be resource-intensive, as it may need to process large volumes of text and use complex event detection models.
*   **Dependencies:** `data_retrieval_agent`
*   **Developer Notes:** The accuracy of this agent can be improved by fine-tuning the event detection model on a domain-specific dataset.

---

## `technical_analyst_agent`

*   **File:** `core/agents/technical_analyst_agent.py`
*   **Description:** This agent is responsible for performing technical analysis of securities. It can analyze price charts, identify technical indicators, and generate trading signals.
*   **Configuration:** `config/agents.yaml`
    *   `technical_analysis_indicators`: A list of technical analysis indicators to use.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.Agent`.
*   **Agent Forge and Lifecycle:** This agent is created by the Agent Forge on demand and runs until it has completed its technical analysis task.
*   **Model Context Protocol (MCP):** The agent maintains a list of the trading signals that it has generated.
*   **Tools and Hooks:**
    *   **Tools:** `technical_analysis_tool`
    *   **Hooks:** `pre_analysis_hook`, `post_analysis_hook`
*   **Compute and Resource Requirements:** This agent is not very resource-intensive.
*   **Dependencies:** `market_data_api`
*   **Developer Notes:** This agent can be extended by adding new technical analysis indicators.

---

## `red_team_agent`

*   **File:** `core/agents/red_team_agent.py`
*   **Description:** The Red Team Agent acts as an adversary to the system. It generates novel and challenging scenarios (stress tests) to validate risk models and system resilience.
*   **Configuration:** `config/agents.yaml`
    *   `num_scenarios`: Number of scenarios to generate.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.AgentBase`.
*   **Agent Forge and Lifecycle:** Created on demand or during stress test cycles.
*   **Model Context Protocol (MCP):** Maintains context of generated scenarios and their severity.
*   **Tools and Hooks:** None currently.
*   **Compute and Resource Requirements:** Low to Medium.
*   **Dependencies:** None.

---

## `meta_cognitive_agent`

*   **File:** `core/agents/meta_cognitive_agent.py`
*   **Description:** Responsible for monitoring the reasoning and outputs of other agents to ensure logical consistency, coherence, and alignment with core principles.
*   **Configuration:** `config/agents.yaml`
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.AgentBase`.
*   **Agent Forge and Lifecycle:** Continuous monitoring or per-task review.
*   **Model Context Protocol (MCP):** Stateless reviews.
*   **Tools and Hooks:** None currently.
*   **Compute and Resource Requirements:** Low.
*   **Dependencies:** None.

---

## `reflector_agent`

*   **File:** `core/agents/reflector_agent.py`
*   **Description:** A simple agent that reflects its input back to the sender. Useful for testing agent routing and cyclical reasoning.
*   **Configuration:** None.
*   **Architecture and Base Agent:** Inherits from `core.system.v22_async.async_agent_base.AsyncAgentBase`.
*   **Agent Forge and Lifecycle:** On demand for testing.
*   **Model Context Protocol (MCP):** Stateless.
*   **Tools and Hooks:** None.
*   **Compute and Resource Requirements:** Negligible.
*   **Dependencies:** None.

---

## `behavioral_economics_agent`

*   **File:** `core/agents/behavioral_economics_agent.py`
*   **Description:** Analyzes market data and user interactions for signs of cognitive biases and irrational behavior.
*   **Configuration:** `config/agents.yaml`
    *   `bias_patterns`: Dictionary of bias patterns to look for.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.AgentBase`.
*   **Agent Forge and Lifecycle:** Continuous or on demand.
*   **Model Context Protocol (MCP):** Stateless.
*   **Tools and Hooks:** Uses `transformers` pipeline for sentiment analysis.
*   **Compute and Resource Requirements:** Medium (uses ML model).
*   **Dependencies:** `transformers`, `torch`.

---

## `cyclical_reasoning_agent`

*   **File:** `core/agents/cyclical_reasoning_agent.py`
*   **Description:** An agent capable of cyclical reasoning, routing its output back to itself or other agents for iterative improvement.
*   **Configuration:** None.
*   **Architecture and Base Agent:** Inherits from `core.system.v22_async.async_agent_base.AsyncAgentBase`.
*   **Agent Forge and Lifecycle:** Task-based.
*   **Model Context Protocol (MCP):** Tracks iteration count.
*   **Tools and Hooks:** None.
*   **Compute and Resource Requirements:** Low.
*   **Dependencies:** None.

---

## `prompt_generation_agent`

*   **File:** `core/agents/prompt_generation_agent.py`
*   **Description:** An agent that generates a high-quality prompt from a user query.
*   **Configuration:** None.
*   **Architecture and Base Agent:** Inherits from `core.agents.agent_base.AgentBase`.
*   **Agent Forge and Lifecycle:** On demand.
*   **Model Context Protocol (MCP):** Stateless.
*   **Tools and Hooks:** Uses Semantic Kernel or LLM.
*   **Compute and Resource Requirements:** Low.
*   **Dependencies:** `semantic_kernel` (optional).
