# ADAM Technical Specification Guides

This document is the central hub for the technical specifications of the ADAM platform.

---

## ADAM v20.0 - Master Technical Design Specification

The canonical technical specification for the Adam v20.0 system upgrade, focusing on Enhanced Autonomy, Causal Inference, and Generative Simulation.

*   **[Master Technical Design Specification](./Adam_v20.0_TECHNICAL_SPECIFICATION.md)**

### v20.0 Schemas and Ontologies
*   **[Agent Proposal Standard (APS/1.0)](./schemas/agent_proposal.schema.json):** A JSON Schema for system-generated agent proposals.
*   **["Black Swan" Scenario Definition Schema (BSSDS/1.0)](./schemas/black_swan_scenario.schema.yaml):** A YAML schema for defining simulation scenarios.
*   **[Adam Causal Predicate Set (ACPS/1.0)](./ontologies/acps.ttl):** An RDF/OWL ontology for representing causal relationships.

---

# ADAM v21.0 - Technical Specification Guide

## Introduction

This document is the central hub for the technical specification of the ADAM v21.0 platform. It provides a comprehensive overview of the system, its architecture, and its components. This guide is intended for developers, architects, and other technical stakeholders who are involved in the development, deployment, and maintenance of the system.

## Table of Contents

This guide is organized into the following sections:

1.  **[Project Vision and Business Requirements](./PROJECT_VISION.md):** Outlines the project's vision, goals, and key business requirements.
2.  **[System Architecture and Design](./ARCHITECTURE.md):** Describes the system's components, their interactions, and the overall design.
3.  **[Data Strategy and Management](./DATA_STRATEGY.md):** Details the data storage strategy, including the data warehouse, SharePoint integration, and the process for ingesting and labeling data.
4.  **[API Specification](./api_specification.yaml):** A formal OpenAPI (Swagger) specification for the central API.
5.  **[User Interface and Chatbot](./UI_AND_CHATBOT.md):** Describes the user interface and the functionality of the chatbot.
6.  **[Agentic Processes and Human-in-the-Loop](./AGENTIC_PROCESSES.md):** Describes the agentic processes, including the different modes of operation and the mechanisms for human-in-the-loop interaction.
7.  **[Resource Management and Tracking](./RESOURCE_MANAGEMENT.md):** Explains how compute and token usage will be tracked and managed.
8.  **[Setup and Deployment Guide](./SETUP_AND_DEPLOYMENT.md):** A comprehensive guide for setting up and deploying the system.
9.  **[Prompt Library Guide](./PROMPT_LIBRARY_GUIDE.md):** A guide for using and extending the prompt library.

## Additional Documents

*   **[Security Specification](./SECURITY.md):** Details the security measures for the system.
*   **[Testing Strategy](./TESTING_STRATEGY.md):** Outlines the strategy for testing the system.
*   **[Glossary](./GLOSSARY.md):** A glossary of key terms and concepts used in the technical specification.

## Supporting Files

The following files are also part of this technical specification:

*   **[FIBO-Aligned Schema](./schema_fibo.yaml):** The YAML definition of the FIBO-aligned schema for the Financial Digital Twin.
*   **[Sample Configuration](./config.sample.json):** A sample configuration file for the system.
*   **[Deployment Script](./deploy.sh):** A shell script to automate the setup and deployment process.

## Project README

For the general project README, please see [../README_PROJECT.md](../README_PROJECT.md).

