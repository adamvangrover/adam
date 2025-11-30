# Financial Digital Twin Framework

This directory contains the foundational components for the Financial Digital Twin, a next-generation intelligence platform for lending operations.

## Purpose

The goal of this framework is to provide a structured, code-based representation of the system's core components. This includes the semantic ontology, knowledge graph schemas, AI agent definitions, and supporting code.

---

## Schema Strategy: A Dual Approach

This project employs a dual-schema strategy to balance rapid development with long-term strategic alignment. Two parallel schemas co-exist within this directory: a legacy/custom schema and the strategic, FIBO-aligned schema.

### 1. Strategic Schema (FIBO-Aligned)

This is the official, enterprise-grade data model for the Financial Digital Twin. It is based on the **Financial Industry Business Ontology (FIBO)** to ensure semantic interoperability and conceptual soundness. All new, core platform development should adhere to this model.

*   **`ontology.md`**: The primary source of truth. This document provides a detailed overview of the FIBO-aligned ontology, the mapping of business concepts to FIBO classes, and the governance framework for extending the ontology. **All developers should start here.**
*   **`schema_fibo.py`**: A Python dataclass implementation of the FIBO-aligned schema. This provides a concrete, code-based reference for data ingestion, validation, and application development.

### 2. Legacy Schema (Custom)

This schema was developed prior to the adoption of the formal FIBO standard. It remains to support existing applications and to serve as a reference or a basis for rapid prototyping of non-critical features.

*   **`schema.py`**: A Python dataclass implementation of the custom schema.
*   **`schema.cypher`**: A Cypher script for applying constraints for the custom schema in a Neo4j database.
*   **`schema.yaml`**: A YAML representation of the custom schema.

---

## Other Components

*   **`nexus_agent.py`**: The core AI agent for interacting with the digital twin.
*   **`influxdb_client.py`**: A client for interacting with the time-series database (InfluxDB).
*   **`prompts.md`**: A library of specialized prompts for AI agents.
*   **`_legacy/`**: This directory contains older, deprecated files that have been archived for historical reference.
