# Agent Documentation: [Agent Name]

## 1. Purpose & Functionality

*   **Primary Goal:** What is the main objective of this agent? What problem does it solve?
*   **Key Functions:** List the specific functions or capabilities of the agent (e.g., fetches data, performs analysis, generates reports).

## 2. Inputs

*   **Data/Objects:** What data or objects does this agent require to operate?
*   **Configuration:** What parameters in `agents.yaml` or other config files does this agent use?
    *   `parameter_name`: (type) Description of the parameter.

## 3. Outputs

*   **Data/Objects:** What data, artifacts, or objects does this agent produce?
*   **Output Schema:** Describe the structure of the output (e.g., JSON schema, class structure).

## 4. Dependencies

*   **Internal Agents:** Which other agents does this agent depend on or interact with?
*   **External Services:** Does this agent rely on any external APIs or data sources?
*   **Libraries:** Are there any special or non-standard Python libraries required?

## 5. Error Handling

*   **Common Errors:** What are the common failure modes?
*   **Recovery:** How does the agent handle these errors? Does it retry, fail gracefully, or produce a specific error message?

## 6. How to Use

*   **Example Workflow:** Provide a brief example of how to configure and run this agent within a workflow.
*   **Example Configuration (`agents.yaml`):**
    ```yaml
    - name: [Agent Name]
      class: [Agent Class Path]
      # Add other parameters here
    ```