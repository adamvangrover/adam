# Virtual Twin Schema Documentation

This document provides a detailed explanation of the `virtual_twin_schema.json`, which is the central configuration artifact for defining and instantiating a Financial Digital Twin within the ADAM ecosystem.

## 1. Top-Level Properties

The root of the schema defines the fundamental characteristics of the Virtual Twin.

| Property      | Type   | Description                                                                                                                              | Required |
|---------------|--------|------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `id`          | String | A unique, machine-readable identifier for the twin instance (e.g., `acme_lending_operations_v1`). This is the primary key for the twin.    | Yes      |
| `version`     | String | The semantic version (`MAJOR.MINOR.PATCH`) of this twin definition file. This allows for versioning and evolution of the twin's configuration. | Yes      |
| `name`        | String | A human-readable name for the twin (e.g., "ACME Lending Operations").                                                                    | Yes      |
| `description` | String | A brief narrative describing the purpose and scope of this Virtual Twin.                                                                 | No       |
| `ontology`    | Object | Defines the semantic core of the twin, based on the FIBO standard. See section 2.                                                        | Yes      |
| `datasources` | Array  | A list of data sources that provide the raw data to populate the twin's knowledge graph. See section 3.                                  | No       |
| `agents`      | Array  | A list of AI agents that interact with or manage the twin. See section 4.                                                                | No       |
| `visualizations`| Array| A list of analytical outputs, like dashboards or maps, derived from the twin. See section 5.                                             | No       |

---

## 2. The `ontology` Object

This object specifies the formal ontology that provides the conceptual model for the twin's knowledge graph. Adherence to a shared ontology like FIBO is critical for ensuring data consistency and enabling complex relationship analysis.

| Property     | Type  | Description                                                                                                                               | Required |
|--------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `uri`        | String (URI) | The URI pointing to the core ontology definition (e.g., the official TTL file for FIBO). This forms the semantic backbone of the twin. | Yes      |
| `extensions` | Array of Strings (URI) | A list of URIs pointing to proprietary ontology extensions. This allows for customizing the core ontology to fit specific business needs while maintaining a clear separation from the standard. | No       |

---

## 3. The `datasources` Array

This array defines the set of data providers that feed information into the Virtual Twin. Each object in the array represents a single, configurable data source.

| Property | Type   | Description                                                                                                                            |
|----------|--------|----------------------------------------------------------------------------------------------------------------------------------------|
| `name`   | String | A unique name for the data source (e.g., `internal_loan_database`).                                                                    |
| `type`   | String | The type of the data source. Must be one of `api`, `database`, or `file_stream`.                                                        |
| `config` | Object | A flexible object containing the specific configuration parameters for the source (e.g., connection strings, API keys, file paths).      |

---

## 4. The `agents` Array

This array lists the AI agents that are responsible for managing, analyzing, and interacting with the Virtual Twin. Each agent has a specific role and is activated by defined triggers.

| Property   | Type   | Description                                                                                                                            |
|------------|--------|----------------------------------------------------------------------------------------------------------------------------------------|
| `name`     | String | The class name of the agent to be instantiated (e.g., `IngestionAgent`, `NexusQueryAgent`).                                            |
| `role`     | String | A human-readable description of the agent's function within the twin ecosystem (e.g., "Processes incoming loan applications").         |
| `triggers` | Object | An object defining the conditions that activate the agent. This could be a schedule (e.g., cron expression), a new data event, or a direct user command. |

---

## 5. The `visualizations` Array

This array defines the user-facing analytical tools and outputs that are generated from the Virtual Twin's data.

| Property      | Type   | Description                                                                                                                            |
|---------------|--------|----------------------------------------------------------------------------------------------------------------------------------------|
| `name`        | String | The display name of the visualization (e.g., "Real-time Credit Contagion Map").                                                        |
| `type`        | String | The type of the visualization. Must be one of `realtime_probability_map` or `dashboard`.                                               |
| `query_agent` | String | The name of the agent responsible for executing the queries and generating the data required for this specific visualization.            |
