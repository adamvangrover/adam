# Agent Development Guide

This document provides a comprehensive guide for developers creating new agents for the ADAM system. It covers the agent development workflow, best practices, debugging and testing, and the agent API. For a catalog of existing agents, see the [Agent Catalog](AGENT_CATALOG.md).

## 1. Agent Development Workflow

The agent development workflow consists of the following steps:

1.  **Define the agent's role and responsibilities.** The first step is to clearly define the agent's role and responsibilities. What is the agent's purpose? What tasks will it perform? What data will it need?
2.  **Design the agent's architecture.** Once you have defined the agent's role and responsibilities, you can start to design its architecture. What will be the agent's main components? How will they interact with each other? What will be the agent's inputs and outputs?
3.  **Implement the agent.** The next step is to implement the agent in Python. You will need to create a new class that inherits from the `Agent` class in `agent_base.py`.
4.  **Test the agent.** Once you have implemented the agent, you will need to test it thoroughly. This includes writing unit tests, integration tests, and end-to-end tests.
5.  **Deploy the agent.** Once the agent has been tested and is working correctly, you can deploy it to the ADAM system.

## 2. Best Practices

When developing new agents, it is important to follow these best practices:

*   **Keep agents small and focused.** Each agent should have a single responsibility. This will make the agents easier to understand, test, and maintain.
*   **Use the message-passing system for all communication between agents.** This will make your code more modular and easier to test.
*   **Use the knowledge base to store and share information.** This will make your code more reusable and easier to maintain.
*   **Write unit tests for all of your code.** This will help you to ensure that your code is working correctly and that it is easy to refactor.
*   **Use a consistent coding style.** This will make your code easier to read and understand.
*   **Document your code.** This will make it easier for other developers to understand and to use your code.

## 3. Debugging and Testing

The ADAM system provides a number of tools for debugging and testing agents.

### 3.1. `echo_agent`

The `echo_agent` is a simple agent that echoes back any message it receives. It can be used to test the message-passing system and to debug communication problems between agents.

### 3.2. Standalone Mode

Agents can be run in a standalone mode for testing and debugging purposes. To run an agent in standalone mode, you can use the `scripts/run_agent.py` script. This script takes the name of the agent as a command-line argument and starts the agent in its own process.

### 3.3. Unit Testing

Unit tests are used to test individual units of code, such as functions and classes. Unit tests are written using the `pytest` framework and are located in the `tests/unit` directory.

### 3.4. Integration Testing

Integration tests are used to test the interactions between different components of the system. Integration tests are written using the `pytest` framework and are located in the `tests/integration` directory.

### 3.5. End-to-End Testing

End-to-end tests are used to test the entire system from start to finish. End-to-end tests are written using a combination of `pytest` and other tools, such as `Selenium` and `Behave`. End-to-end tests are located in the `tests/e2e` directory.

## 4. Agent API Reference

The `Agent` class in `agent_base.py` provides the following methods and properties:

*   **`__init__(self, name, persona)`:** Initializes the agent with a name and persona.

    *   **Code Example:**
        ```python
        from core.agents.agent_base import Agent

        class MyAgent(Agent):
            def __init__(self):
                super().__init__("MyAgent", "A friendly agent that helps with tasks.")
        ```

*   **`run(self)`:** The main entry point for the agent. This method is called by the system to start the agent's execution.

    *   **Code Example:**
        ```python
        from core.agents.agent_base import Agent

        class MyAgent(Agent):
            def __init__(self):
                super().__init__("MyAgent", "A friendly agent that helps with tasks.")

            def run(self):
                self.log("MyAgent is running.")
                # Agent's main logic goes here
        ```

*   **`send_message(self, recipient, message)`:** Sends a message to another agent.

    *   **Code Example:**
        ```python
        self.send_message("OtherAgent", "Hello from MyAgent!")
        ```

*   **`receive_message(self, sender, message)`:** Receives a message from another agent.

    *   **Code Example:**
        ```python
        def receive_message(self, sender, message):
            self.log(f"Received message from {sender}: {message}")
        ```

*   **`get_knowledge(self, query)`:** Retrieves information from the knowledge base. For more information on the knowledge base, see the [Data Navigation Guide](../../data/DATA_NAVIGATION.md).

    *   **Code Example:**
        ```python
        company_info = self.get_knowledge("What is the stock price of GOOGL?")
        ```

*   **`log(self, message)`:** Logs a message to the system's log file.

    *   **Code Example:**
        ```python
        self.log("This is a log message.")
        ```

## 5. Tutorials

### 5.1. "Hello, World" Agent

This tutorial shows how to create a simple "Hello, World" agent.

1.  Create a new file in the `core/agents` directory called `hello_world_agent.py`.
2.  Add the following code to the file:

    ```python
    from core.agents.agent_base import Agent

    class HelloWorldAgent(Agent):
        def __init__(self):
            super().__init__("HelloWorldAgent", "An agent that prints 'Hello, World!'")

        def run(self):
            self.log("Hello, World!")
    ```

3.  Run the agent using the `scripts/run_agent.py` script:

    ```bash
    python scripts/run_agent.py HelloWorldAgent
    ```

### 5.2. Agent that Uses the Knowledge Base

This tutorial shows how to create an agent that uses the knowledge base.

1.  Create a new file in the `core/agents` directory called `knowledge_agent.py`.
2.  Add the following code to the file:

    ```python
    from core.agents.agent_base import Agent

    class KnowledgeAgent(Agent):
        def __init__(self):
            super().__init__("KnowledgeAgent", "An agent that uses the knowledge base.")

        def run(self):
            knowledge = self.get_knowledge("What is the capital of France?")
            self.log(f"The capital of France is: {knowledge}")
    ```

### 5.3. Agent that Communicates with Another Agent

This tutorial shows how to create two agents that communicate with each other.

1.  Create a new file in the `core/agents` directory called `sender_agent.py`.
2.  Add the following code to the file:

    ```python
    from core.agents.agent_base import Agent

    class SenderAgent(Agent):
        def __init__(self):
            super().__init__("SenderAgent", "An agent that sends a message.")

        def run(self):
            self.send_message("ReceiverAgent", "Hello from SenderAgent!")
    ```

3.  Create a new file in the `core/agents` directory called `receiver_agent.py`.
4.  Add the following code to the file:

    ```python
    from core.agents.agent_base import Agent

    class ReceiverAgent(Agent):
        def __init__(self):
            super().__init__("ReceiverAgent", "An agent that receives a message.")

        def receive_message(self, sender, message):
            self.log(f"Received message from {sender}: {message}")
    ```

## 6. Advanced Topics

### 6.1. Agent Lifecycle Management

The agent lifecycle is managed by the `AgentOrchestrator`. The orchestrator is responsible for creating, starting, stopping, and deleting agents.

### 6.2. Error Handling

Agents should use try-except blocks to handle errors gracefully. The `ErrorHandler` class can be used to log and report errors.

### 6.3. Performance Tuning

The performance of an agent can be tuned by optimizing its code and by using caching. The `CacheManager` can be used to cache frequently accessed data.

## 7. Developer Notes

*   The `agent_base.py` file contains the base class for all agents.
*   The `agent_orchestrator.py` file contains the code for managing the agent lifecycle.
*   The `knowledge_base.py` file contains the code for accessing the knowledge base.

## 8. Future Development

*   **Agent Sandboxing:** In the future, we plan to implement agent sandboxing to provide a more secure environment for running agents.
*   **Agent Marketplace:** We also plan to create an agent marketplace where developers can share and sell their agents.

By following this guide, you can help to ensure that your agents are well-designed, easy to maintain, and work together effectively with other agents in the ADAM system.
