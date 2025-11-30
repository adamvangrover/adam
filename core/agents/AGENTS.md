# Agents

This directory contains the autonomous agents that are the heart of the ADAM system. Each agent is a specialized AI module responsible for a specific aspect of financial analysis, risk assessment, or knowledge management.

## Core Capabilities

Agents in the ADAM system possess a range of core capabilities that enable them to perform their tasks effectively:

*   **Data Processing:** Agents can process a wide variety of data types, including structured data (e.g., CSV, JSON), unstructured data (e.g., text, images), and semi-structured data (e.g., HTML, XML).
*   **Natural Language Understanding (NLU):** Agents use NLU to understand and interpret human language, allowing them to process text-based data sources and interact with users.
*   **Natural Language Generation (NLG):** Agents use NLG to generate human-readable text, such as reports, summaries, and chat messages.
*   **Decision-Making:** Agents use a variety of decision-making techniques, including rule-based systems, machine learning models, and optimization algorithms, to make informed decisions.
*   **Learning:** Agents can learn from experience and adapt their behavior over time. This includes both supervised and unsupervised learning.

## Runtimes and Execution

The ADAM system provides a robust runtime environment for executing agents. The runtime is responsible for managing the lifecycle of the agents, providing them with the resources they need to operate, and ensuring the overall stability of the system.

### Main Loop

The main loop is the heart of the agent runtime. It is responsible for iterating through the active agents and giving each agent an opportunity to execute. The main loop also handles a variety of other tasks, such as processing messages, managing the agent lifecycle, and monitoring the health of the system.

### Threading Model

The agent runtime uses a multi-threaded architecture to allow multiple agents to execute concurrently. Each agent runs in its own thread, which allows it to perform long-running tasks without blocking the main loop.

### Resource Management

The agent runtime is responsible for managing the resources that are used by the agents, such as CPU, memory, and network bandwidth. The runtime uses a variety of techniques to ensure that resources are allocated fairly and efficiently, and to prevent any single agent from consuming too many resources.

## Implementation Details

The agents in the ADAM system are implemented using a combination of object-oriented programming and design patterns.

### Base Agent Class

All agents inherit from the `Agent` class in `agent_base.py`. This class provides the basic structure for all agents, including methods for sending and receiving messages, accessing the knowledge base, and logging events.

### Inheritance Patterns

The ADAM system uses a variety of inheritance patterns to create specialized agents. For example, the `FinancialAnalystAgent` class inherits from the `Agent` class and adds a variety of methods for performing financial analysis.

### Design Patterns

The ADAM system uses a variety of design patterns to improve the modularity, extensibility, and maintainability of the code. These include the Singleton pattern for managing global resources, the Factory pattern for creating new agents, and the Observer pattern for handling events.

## Standalone Operation

Agents can be run in a standalone mode for testing and debugging purposes. To run an agent in standalone mode, you can use the `scripts/run_agent.py` script. This script takes the name of the agent as a command-line argument and starts the agent in its own process.

```bash
python scripts/run_agent.py --agent-name market_sentiment_agent
```

### Agent Lifecycle

The lifecycle of an agent in the ADAM system is managed by the agent runtime. The lifecycle consists of the following stages:

1.  **Initialization:** When an agent is first loaded, it is initialized with its configuration. This includes setting up any necessary resources, such as database connections or API clients.
2.  **Execution:** Once an agent is initialized, it enters the execution stage. In this stage, the agent's `execute` method is called repeatedly by the main loop. The `execute` method contains the core logic of the agent and is responsible for performing the agent's tasks.
3.  **Termination:** When an agent is no longer needed, it is terminated. This includes releasing any resources that were acquired during initialization.

### Main Loop

The main loop is the heart of the agent runtime. It is responsible for iterating through the active agents and giving each agent an opportunity to execute. The main loop also handles a variety of other tasks, such as processing messages, managing the agent lifecycle, and monitoring the health of the system.

### Threading Model

The agent runtime uses a multi-threaded architecture to allow multiple agents to execute concurrently. Each agent runs in its own thread, which allows it to perform long-running tasks without blocking the main loop.

### Resource Management

The agent runtime is responsible for managing the resources that are used by the agents, such as CPU, memory, and network bandwidth. The runtime uses a variety of techniques to ensure that resources are allocated fairly and efficiently, and to prevent any single agent from consuming too many resources.

## Implementation Details

The agents in the ADAM system are implemented using a combination of object-oriented programming and design patterns.

### Base Agent Class

All agents inherit from the `Agent` class in `agent_base.py`. This class provides the basic structure for all agents, including methods for sending and receiving messages, accessing the knowledge base, and logging events.

### Inheritance Patterns

The ADAM system uses a variety of inheritance patterns to create specialized agents. For example, the `FinancialAnalystAgent` class inherits from the `Agent` class and adds a variety of methods for performing financial analysis.

### Design Patterns

The ADAM system uses a variety of design patterns to improve the modularity, extensibility, and maintainability of the code. These include the Singleton pattern for managing global resources, the Factory pattern for creating new agents, and the Observer pattern for handling events.

## Creating and Using Agents

This section provides a more comprehensive example of how to create and use an agent in the ADAM system.

### Defining an Agent

To define an agent, you need to create a new class that inherits from the `Agent` class in `agent_base.py`. The following code snippet shows how to define a simple `EchoAgent` that echoes back any message it receives:

```python
from core.agents.agent_base import Agent

class EchoAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

    def execute(self, message):
        return message
```

### Implementing the `execute` Method

The `execute` method is the main entry point for the agent. It is called repeatedly by the main loop and is responsible for performing the agent's tasks. The `execute` method takes a message as input and should return a message as output.

### Running an Agent in Standalone Mode

Agents can be run in a standalone mode for testing and debugging purposes. To run an agent in standalone mode, you can use the `scripts/run_agent.py` script. This script takes the name of the agent as a command-line argument and starts the agent in its own process.

```bash
python scripts/run_agent.py --agent-name echo_agent
```

When you run this command, the `EchoAgent` will be started and will wait for messages. You can then send messages to the agent and it will echo them back to you.

## Comprehensive Agent Profiles

Here are in-depth profiles of some of the key agents in the ADAM system:

### `fundamental_analyst_agent`

*   **Role:** Performs fundamental analysis of companies.
*   **Responsibilities:**
    *   Retrieves financial data from a variety of sources.
    *   Analyzes financial statements, such as the balance sheet, income statement, and cash flow statement.
    *   Calculates key financial ratios, such as the price-to-earnings ratio and the debt-to-equity ratio.
    *   Generates reports on the financial health of companies.
*   **Configuration:**
    *   `data_sources`: A list of data sources to use for financial data.
    *   `analysis_modules`: A list of analysis modules to use for financial analysis.

### `market_sentiment_agent`

*   **Role:** Gauges market sentiment from a variety of sources.
*   **Responsibilities:**
    *   Retrieves news articles, social media posts, and other text-based data sources.
    *   Uses natural language processing to analyze the sentiment of the text.
    *   Generates reports on market sentiment.
*   **Configuration:**
    *   `data_sources`: A list of data sources to use for sentiment analysis.
    *   `sentiment_analysis_model`: The name of the sentiment analysis model to use.

## Mission Critical Procedures (MCP)

The ADAM system has a set of mission-critical procedures (MCPs) that are designed to ensure the stability and reliability of the system. These procedures are executed in response to critical events, such as a system crash or a network failure.

The MCPs include:

*   **System Restart:** Restarts the system in a safe mode.
*   **Data Recovery:** Recovers data from a backup.
*   **Failover:** Switches to a backup system.

## Agent-to-Agent (A2A) Communication

Agents in the ADAM system communicate with each other through a message-passing system. This allows agents to collaborate on tasks and share information.

### Message Schemas

Messages are sent as JSON objects with a predefined schema. The schema defines the structure of the message and the types of the fields in the message.

### Interaction Patterns

There are several standard interaction patterns that are used for common interactions between agents:

*   **Request-Response:** One agent sends a request to another agent and waits for a response.
*   **Publish-Subscribe:** One agent publishes a message to a topic, and any interested agents can subscribe to that topic to receive the message.
*   **Broadcast:** One agent sends a message to all other agents in the system.

## Shared Context and State Management

Agents in the ADAM system can share context and manage state through a variety of mechanisms.
Effective state management is crucial for building robust and reliable agents. Agents in the ADAM system can share context and manage state through a variety of mechanisms.

### In-Memory State

For simple use cases, agents can store state in memory using instance variables. This is the simplest and fastest way to manage state, but it is not persistent and is not suitable for sharing state between multiple instances of the same agent.

### Shared Memory

Agents can use shared memory to share data with other agents on the same machine. This is a fast and efficient way to share data, but it is not suitable for sharing data between agents on different machines.

### Distributed Caches

Agents can use a distributed cache, such as Redis or Memcached, to share data with other agents in a distributed environment. This is a more scalable and resilient way to share data than shared memory.
For more complex use cases, agents can use a distributed cache, such as Redis or Memcached, to store and share state. Distributed caches provide a scalable and resilient way to manage state, and they can be used to share state between multiple instances of the same agent or between different agents.

### Database

For long-term persistence, agents can use a database to store state. The ADAM system provides a database abstraction layer that makes it easy to use a variety of databases, such as PostgreSQL, MySQL, and SQLite.

## Error Handling

Robust error handling is essential for building a reliable and resilient agent-based system. Agents in the ADAM system should be designed to handle a variety of errors, such as network failures, API errors, and invalid data.

### Timeouts

When making requests to external services, agents should use timeouts to prevent them from blocking indefinitely. Timeouts can be configured in the `config/api.yaml` file.

### Retries

For transient errors, such as network failures, agents should use retries to automatically retry the request. The number of retries and the delay between retries can be configured in the `config/api.yaml` file.

### Circuit Breakers

For more serious errors, such as an API that is consistently failing, agents should use a circuit breaker to prevent them from repeatedly making requests to the failing service. The circuit breaker will automatically open after a certain number of failures and will remain open for a configurable period of time.

### Fallbacks

When a request fails, agents should have a fallback mechanism in place. This could be as simple as returning a default value or as complex as using a different data source.

## Semantic Kernel Integration

The ADAM system is integrated with the Semantic Kernel, a library that provides a set of tools for building advanced AI applications. The Semantic Kernel provides a variety of features that are used by the agents in the ADAM system, such as:

*   **Prompt Templating:** A way to create reusable prompts for the LLM.
*   **Function Chaining:** A way to chain together multiple functions to create complex workflows.
*   **Memory and Knowledge Base:** A way to store and retrieve information from a knowledge base.

## Knowledge Base and Knowledge Graph

The ADAM system uses a knowledge base and a knowledge graph to store and manage information.

### Ontology

The knowledge base is based on an ontology that defines the concepts and relationships in the financial domain. The ontology is used to structure the data in the knowledge base and to make it easier for agents to query and reason about the data.

### Data Models

The knowledge base uses a variety of data models to represent the data, including:

*   **Relational Model:** Used to store structured data, such as financial statements.
*   **Graph Model:** Used to store graph-based data, such as social networks and supply chains.
*   **Document Model:** Used to store unstructured data, such as news articles and research reports.

### Query Languages

Agents can use a variety of query languages to query the knowledge base, including:

*   **SQL:** Used to query the relational data.
*   **Cypher:** Used to query the graph data.
*   **SPARQL:** Used to query the RDF data.

## Developer Notes

Here are some notes and best practices for developers working on the agent-based system:

*   **Keep agents small and focused.** Each agent should have a single responsibility.
*   **Use the message-passing system for all communication between agents.** This will make your code more modular and easier to test.
*   **Use the knowledge base to store and share information.** This will make your code more reusable and easier to maintain.
*   **Write unit tests for all of your code.** This will help you to ensure that your code is working correctly and that it is easy to refactor.

## Future Development

Here is a roadmap for the future development of the agent-based system:

*   **Add more agents.** We plan to add more agents to the system to cover a wider range of financial analysis tasks.
*   **Improve the A2A communication protocols.** We plan to improve the A2A communication protocols to make them more efficient and reliable.
*   **Enhance the knowledge base.** We plan to enhance the knowledge base by adding more data sources and improving the ontology.
*   **Develop a user interface.** We plan to develop a user interface that will allow users to interact with the agents and to visualize the results of their analysis.
