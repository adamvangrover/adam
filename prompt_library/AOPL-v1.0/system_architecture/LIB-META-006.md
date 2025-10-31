# LIB-META-006: System Documentation Generator

*   **ID:** `LIB-META-006`
*   **Version:** `1.0`
*   **Author:** Jules
*   **Objective:** To generate clear, comprehensive, and user-friendly documentation for a complex AI system or agentic workflow, based on the architectural design.
*   **When to Use:** After designing a new agentic system (using `LIB-META-001`), use this prompt to create the initial `README.md` or internal wiki page for the project. This ensures documentation keeps pace with design.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[System_Name]`: The name of the AI system or agent.
    *   `[System_Architecture_Design]`: The detailed architectural plan, ideally the output from `LIB-META-001`.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `DocumentationAgent` or `SystemArchitectAgent`.
    *   **Automated Documentation:** This prompt can be integrated into a CI/CD pipeline for agent development. After an agent's design is approved, this prompt can be automatically run to generate or update its documentation, ensuring it's never out of date.

---

### **Example Usage**

```
[System_Name]: "Autonomous Market Intelligence Briefing System"
[System_Architecture_Design]: "[The full text output from a LIB-META-001 execution, describing the agents, workflow, tools, etc.]"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Principal Technical Writer

# CONTEXT:
You are an expert technical writer. Your skill is to take a complex system architecture design and transform it into clear, concise, and easy-to-understand documentation. The documentation should be suitable for both technical and semi-technical audiences.

# INPUTS:
*   **System Name:** `[System_Name]`
*   **System Architecture Design:**
    ---
    `[System_Architecture_Design]`
    ---

# TASK:
Generate a comprehensive `README.md` file for the specified system. The documentation must be well-structured and cover all key aspects of the system.

---
# **README.md: [System_Name]**

## **1. Overview**
*(Write a one-paragraph summary of the system's purpose. What business problem does it solve? What is its primary function?)*

## **2. System Architecture**
*(Summarize the key components of the architecture provided.)*
*   **Workflow Style:** (e.g., "This system uses a sequential pipeline model...")
*   **Orchestrator:** (e.g., "The `OrchestratorAgent` is responsible for managing the workflow.")

## **3. The Agents**
*(Provide a table describing the agents involved in the system.)*

| Agent Name | Role & Responsibilities | Key Skills / Prompts Used |
| :--- | :--- | :--- |
| [Agent 1 Name] | [Description] | [e.g., `LIB-PRO-002`] |
| [Agent 2 Name] | [Description] | [e.g., `LIB-PRO-001`] |
| ... | ... | ... |

## **4. Workflow & Data Flow**
*(Describe the step-by-step process of how the system operates. Use a numbered list.)*
1.  **Trigger:** The process begins when [describe the trigger].
2.  **Step 1:** The `OrchestratorAgent` passes the task to the `[Agent 1 Name]`.
3.  **Step 2:** The `[Agent 1 Name]` creates an artifact called `[Artifact_Name]`.
4.  ...and so on.
5.  **Final Output:** The final result is a `[Final_Output_Type]` which is delivered to `[Destination]`.

## **5. Required Tools & Services**
*(List the external tools or APIs that the system depends on.)*
*   `[Tool_1_Name]`
*   `[Tool_2_Name]`

## **6. How to Use**
*(Provide a simple, clear example of how to run or trigger the system.)*
*   **Triggering the System:**
    ```bash
    [Example command or API call]
    ```

## **7. Human-in-the-Loop**
*(Describe any points in the process that require human intervention or approval.)*

---
```
