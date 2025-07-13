# Core Components

This directory contains the core components of the ADAM system. These components provide the fundamental building blocks for creating and running autonomous agents.

## Subdirectories

*   **`agents/`:** This directory contains the autonomous agents that perform specific tasks.
*   **`analysis/`:** This directory contains modules for performing various types of analysis, such as fundamental and technical analysis.
*   **`data_access/`:** This directory contains modules for accessing data from various sources.
*   **`data_sources/`:** This directory contains modules for specific data sources, such as APIs and databases.
*   **`embeddings/`:** This directory contains modules for creating and managing embeddings.
*   **`llm/`:** This directory contains the language model engine that provides natural language processing capabilities.
*   **`rag/`:** This directory contains the retrieval-augmented generation (RAG) components.
*   **`simulations/`:** This directory contains environments for testing and evaluating the agents' performance.
*   **`system/`:** This directory contains the central infrastructure that supports the agents, including the main loop, data management, and communication.
*   **`tools/`:** This directory contains tools that can be used by the agents.
*   **`utils/`:** This directory contains utility functions that are used throughout the system.
*   **`vectorstore/`:** This directory contains the vector store for storing and retrieving embeddings.
*   **`world_simulation/`:** This directory contains the world simulation components.

## Interacting with Core Components

When interacting with the core components, please adhere to the following principles:

*   **Abstraction:** Interact with the components through their public APIs. Avoid directly accessing their internal implementation details.
*   **Configuration:** Use the configuration files in the `config/` directory to configure the behavior of the core components.
*   **Logging:** Use the logging system to record important events and debug issues.

By following these guidelines, you can help to ensure that the ADAM system remains stable, modular, and easy to maintain.
