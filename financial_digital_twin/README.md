# Financial Intelligence System Framework

This directory contains the foundational components for the Financial Intelligence System, a project designed to create a sophisticated financial analysis platform based on a hybrid knowledge graph and time-series architecture.

## Purpose

The goal of this framework is to provide a structured, code-based representation of the system's core components, including its knowledge graph schema and its library of AI prompts. This allows for version control, collaboration, and automated validation of the system's building blocks.

## Directory Structure

-   `/knowledge_graph`: This directory contains the Python-based definition of the knowledge graph's schema.
    -   `schema.py`: Defines the core nodes (e.g., `Company`, `Loan`) and edges (e.g., `IS_BORROWER_OF`) as Python dataclasses. This serves as the blueprint for the data stored in the graph database.

-   `/prompt_library`: This directory contains a collection of specialized, reusable prompts designed for the AI agents that interact with the system.
    -   `prompts.md`: A Markdown file that lists and describes the prompts for various tasks, such as risk analysis, entity ingestion, and executive summarization.
