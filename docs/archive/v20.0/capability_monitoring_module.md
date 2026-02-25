# Capability Monitoring Module Design Document

## 1. Objective

The primary objective of the Capability Monitoring Module (CMM) is to enhance the autonomy of the Adam system by enabling it to self-diagnose analytical and operational gaps. This module will monitor the performance and interactions of all agents within the system to identify its own limitations, such as frequent task failures, repeated manual interventions, or the inability to process certain data types. Upon identifying a "capability gap," the CMM will initiate a process to propose the creation of a new agent or a modification to an existing one to address the deficiency.

## 2. Lead Agents

*   **Code Alchemist:** Responsible for the underlying code generation and modification required for new or updated agents.
*   **Agent Forge:** Responsible for taking the output of the CMM and structuring it into a formal proposal for a new agent, using a standardized template.

## 3. Architectural Integration

The CMM will be integrated as a core component of the **Agent Orchestrator**. It will operate as a persistent, background process that subscribes to the orchestrator's event bus. This allows it to passively monitor events such as:

*   `task_started`
*   `task_completed`
*   `task_failed`
*   `manual_intervention_required`
*   `data_processing_error`

By hooking into the event bus, the CMM can gather data without adding significant overhead to the primary task execution workflow.

## 4. Data Collection and Monitored Metrics

The CMM will collect and analyze data related to the following key areas:

*   **Task Failures:**
    *   Frequency of failures for specific tasks or agent types.
    *   Error message content and categorization (e.g., data error, API error, logic error).
    *   Correlation of failures with specific data sources or input types.

*   **Manual Interventions:**
    *   Logging every instance where a human-in-the-loop is required to make a decision, correct data, or manually execute a step.
    *   Categorization of the reason for intervention.
    *   Tracking the frequency of interventions in a given workflow.

*   **Unprocessable Data:**
    *   Detecting and logging instances where an agent receives a data type it is not equipped to handle (e.g., geospatial data, unstructured text from a new source).
    *   Analyzing logs for data ingestion errors across all agents.

## 5. Gap Identification Logic

A "capability gap" will be identified based on a set of configurable rules and thresholds. The core logic will be based on pattern detection:

*   **Frequency Threshold:** If a specific task fails more than `X` times in `Y` hours for the same reason, a potential gap is flagged.
*   **Manual Intervention Pattern:** If the same manual intervention point is triggered repeatedly in a workflow, it indicates a process that could be automated.
*   **Novel Data Type:** A single instance of a critical, unprocessable data type can be sufficient to flag a gap.
*   **Cross-Agent Correlation:** If multiple agents fail on the same data source, it points to a systemic gap in data processing capabilities.

When a potential gap is flagged, the CMM will aggregate all relevant data points (logs, error messages, task metadata) into a consolidated "gap report."

## 6. Output and Workflow Trigger

Upon the creation of a "gap report," the CMM's primary output will be to:

1.  **Log the Gap:** Persistently store the gap report in a dedicated database for audit and analysis.
2.  **Trigger Agent Forge:** Send an event to the **Agent Forge** with the gap report as the payload.

The Agent Forge will then parse the report and use its standardized templates to generate a formal, machine-readable proposal for a new agent. This proposal will then be submitted to the human-in-the-loop validation queue for review and approval.
