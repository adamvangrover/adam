# Technical Documentation: Adam Simulation & Prompting Modules

## Overview
This document covers the technical details of the new capabilities introduced in the v23.5 expansion.

## 1. Advanced Prompting: "Prompt-as-Code" v2

The new `GeminiFinancialReportAnalyzer` utilizes a "Prompt-as-Code" paradigm where prompts are not just strings, but structured templates that enforce reasoning.

### Chain-of-Thought (CoT) Implementation
We inject a specific instruction block:
> "Thinking Process: First, silently reason through the text... Then, generate the output..."

This leverages Gemini's ability to generate internal monologues (or we treat the first part of the output as such) to improve logical consistency. In `LLMPlugin`, this is managed via the `thinking_level` parameter.

## 2. Simulation Modules (Forward Looking)

### Crisis Simulation
While currently a graph-based workflow, the v24.0 roadmap moves this to a generative simulation.
*   **Input**: Macro-economic variables (Interest Rates, GDP, Oil Prices).
*   **Engine**: A dedicated `CrisisSimulationAgent` (to be refactored) will use `BigQueryConnector` to pull historical stress-test data.
*   **Output**: A narrative scenario describing the impact on specific portfolios.

## 3. Infrastructure Hooks

### BigQuery Connector
*   **Path**: `core/infrastructure/bigquery_connector.py`
*   **Pattern**: Facade/Adapter.
*   **Usage**: Wraps the `google-cloud-bigquery` library. Currently implements stubs to allow development without live GCP credentials.

### Pub/Sub Connector
*   **Path**: `core/infrastructure/bigquery_connector.py` (shared file for infra).
*   **Usage**: Intended for event-driven architecture. For example, when a new 10-K is dropped into a bucket, a Pub/Sub message triggers the `GeminiFinancialReportAnalyzer`.
