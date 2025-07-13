# Data Navigation Guide

This document provides a high-level overview of the data in the `data` directory and how it is organized. It is intended to help developers navigate the data and to understand how the different data files are related to each other.

## 1. Data Map

The following data map provides a visual representation of the data in the `data` directory and how the different data files are related to each other.

```mermaid
graph TD
    subgraph Knowledge
        A[knowledge_base.json]
        B[knowledge_graph.json]
        C[knowledgegraph.ttl]
    end

    subgraph Decision Trees
        D[credit_rating_decision_tree_v3.json]
    end

    subgraph Ontologies
        E[context_definition.jsonld]
        F[CACM:SaaS_DefaultRisk_v1.jsonld]
    end

    subgraph Core Data
        G[adam_core_data.json]
        H[adam_market_baseline.json]
    end

    subgraph Templates
        I[dcf_model_template.csv]
        J[deal_template.json]
        K[private_company_template.json]
    end

    subgraph Company Data
        L[company_data.json]
    end

    subgraph Risk Data
        M[risk_rating_mapping_v2.json]
    end

    subgraph Training Data
        N[teacher_outputs.jsonl]
        O[sp500_ai_overviews.jsonl]
    end

    A -- "Defines Concepts" --> B
    B -- "Used in" --> D
    E -- "Defines Context for" --> B
    F -- "Defines Context for" --> D
    G -- "Provides Context for" --> L
    H -- "Provides Baseline for" --> L
    I -- "Used for" --> L
    J -- "Used for" --> L
    K -- "Used for" --> L
    M -- "Used in" --> D
    N -- "Used to Train" --> D
    O -- "Used to Train" --> G
```

## 2. Data Dictionary

The following data dictionary provides definitions for all the data fields in the system.

| File | Field | Data Type | Description |
|---|---|---|---|
| `knowledge_base.json` | `Valuation` | object | Contains information about valuation methods, such as DCF and comparables. |
| `knowledge_base.json` | `RiskManagement` | object | Contains information about risk management techniques, such as VaR and credit risk analysis. |
| `knowledge_graph.json` | `nodes` | array | An array of nodes in the knowledge graph. |
| `knowledge_graph.json` | `edges` | array | An array of edges in the knowledge graph. |
| `credit_rating_decision_tree_v3.json` | `tree` | object | The root of the decision tree. |
| `context_definition.jsonld` | `@context` | object | The JSON-LD context for the system. |
| `adam_core_data.json` | `contextual_data` | object | Contains contextual data for the system, such as user profiles and world events. |
| `company_data.json` | `[TICKER]` | object | Contains data for a specific company. |

## 3. Data Lineage

The following diagram shows the lineage of the data in the `data` directory.

```mermaid
graph LR
    subgraph External Sources
        A[Financial APIs]
        B[News Feeds]
        C[Regulatory Filings]
    end

    subgraph Data Processing
        D[Data Ingestion]
        E[Data Cleaning]
        F[Data Transformation]
    end

    subgraph Data Storage
        G[knowledge_base.json]
        H[knowledge_graph.json]
        I[company_data.json]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
```

## 4. Future Development

*   **Data Catalog:** We plan to create a more comprehensive data catalog that will provide more detailed information about the data in the `data` directory.
*   **Data Governance:** We also plan to implement a data governance framework to ensure the quality and consistency of the data.

By providing a clear and comprehensive guide to the data in the `data` directory, we can help developers to more easily navigate and to use the data in their agents.
