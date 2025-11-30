# Financial Digital Twin

This directory contains the foundational components for the Financial Digital Twin, a next-generation intelligence platform for lending operations. This document provides guidance for AI agents working with the Financial Digital Twin codebase.

## High-Level Goal

The primary goal of the Financial Digital Twin is to create a dynamic, virtual representation of a lending business, including its processes, assets, and risks. This allows for advanced analytics, simulations, and AI-driven decision-making.

## Core Components

The Financial Digital Twin is comprised of several key components:

*   **Ontology and Schemas:** These define the structure and meaning of the data in the digital twin.
*   **Nexus Agent:** The core AI agent for interacting with the digital twin.
*   **Time-Series Database:** A database for storing and querying time-series data.
*   **Prompt Library:** A library of specialized prompts for AI agents.

## Dual-Schema Strategy

This project employs a dual-schema strategy to balance rapid development with long-term strategic alignment.

### 1. Strategic Schema (FIBO-Aligned)

This is the official, enterprise-grade data model for the Financial Digital Twin. It is based on the **Financial Industry Business Ontology (FIBO)** to ensure semantic interoperability and conceptual soundness. All new, core platform development should adhere to this model.

*   **`ontology.md`:** The primary source of truth. This document provides a detailed overview of the FIBO-aligned ontology.
*   **`schema_fibo.py`:** A Python dataclass implementation of the FIBO-aligned schema.

### 2. Legacy Schema (Custom)

This schema was developed prior to the adoption of the formal FIBO standard. It remains to support existing applications and to serve as a reference or a basis for rapid prototyping of non-critical features.

*   **`schema.py`:** A Python dataclass implementation of the custom schema.
*   **`schema.cypher`:** A Cypher script for applying constraints for the custom schema in a Neo4j database.
*   **`schema.yaml`:** A YAML representation of the custom schema.

## Working with the Financial Digital Twin

When working with the Financial Digital Twin, please adhere to the following guidelines:

*   **Prioritize the Strategic Schema:** Whenever possible, use the FIBO-aligned schema for new development.
*   **Consult the Ontology:** Before working with the data, please consult the `ontology.md` file to understand the meaning and structure of the data.
*   **Use the Nexus Agent:** The `nexus_agent.py` is the primary interface for interacting with the digital twin.
*   **Leverage the Prompt Library:** The `prompts.md` file contains a library of specialized prompts that can be used to perform common tasks.

By following these guidelines, you can help to ensure that the Financial Digital Twin remains a valuable and reliable resource for the ADAM system.
