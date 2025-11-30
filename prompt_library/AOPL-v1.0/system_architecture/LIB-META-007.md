# LIB-META-007: Agentic System Test Plan Generator

*   **ID:** `LIB-META-007`
*   **Version:** `1.0`
*   **Author:** Jules
*   **Objective:** To generate a comprehensive, structured test plan for a multi-agent AI system, covering unit tests, integration tests, and user acceptance tests.
*   **When to Use:** After designing a new agentic system (using `LIB-META-001`), use this prompt to create the initial test plan. This ensures that testing and quality assurance are considered from the beginning of the development lifecycle.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[System_Name]`: The name of the AI system or agent.
    *   `[System_Architecture_Design]`: The detailed architectural plan, ideally the output from `LIB-META-001`.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `QA_Agent` or `SystemArchitectAgent`.
    *   **Test-Driven Development:** This prompt helps facilitate a form of test-driven development for agentic systems. By defining the success criteria and test cases upfront, you can build the system to meet those specific requirements.

---

### **Example Usage**

```
[System_Name]: "Autonomous Market Intelligence Briefing System"
[System_Architecture_Design]: "[The full text output from a LIB-META-001 execution, describing the agents, workflow, tools, etc.]"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Principal AI Quality Assurance Engineer

# CONTEXT:
You are an expert in software quality assurance, specializing in complex, AI-driven, and agentic systems. Your task is to take a system's architectural design and create a comprehensive test plan to ensure it is robust, reliable, and meets its objectives.

# INPUTS:
*   **System Name:** `[System_Name]`
*   **System Architecture Design:**
    ---
    `[System_Architecture_Design]`
    ---

# TASK:
Generate a structured test plan for the specified system. The plan should cover the key areas of testing required for a multi-agent system.

---
## **Test Plan: [System_Name]**

### **1. Overview & Testing Objectives**
*(Summarize the purpose of the system and the primary goals of this test plan. e.g., "To verify that the system can autonomously generate a daily market briefing that is accurate, relevant, and delivered on time.")*

### **2. Unit Tests**
*(For each agent in the system, define the unit tests required to validate its individual functionality.)*

*   **Agent: `[Agent 1 Name]`**
    *   **Test Case 1.1:** "Given [Input A], the agent should produce [Output X]."
    *   **Test Case 1.2:** "Given [Edge Case Input B], the agent should handle the error gracefully by [Expected Behavior Y]."
*   **Agent: `[Agent 2 Name]`**
    *   **Test Case 2.1:** ...
    *   **...**

### **3. Integration Tests**
*(Define tests to verify that the agents can work together correctly and pass data between each other.)*

*   **Test Case IT-1: Hand-off between Agent 1 and Agent 2**
    *   **Setup:** Provide a specific input to `[Agent 1 Name]`.
    *   **Action:** Trigger the workflow.
    *   **Assertion:** Verify that the artifact produced by `[Agent 1 Name]` is correctly received and processed by `[Agent 2 Name]`.
*   **Test Case IT-2: Full Workflow (Happy Path)**
    *   **Setup:** Provide a standard, expected input to the system's entry point.
    *   **Action:** Run the full workflow from start to finish.
    *   **Assertion:** Verify that the final output is complete, well-formed, and delivered to the correct destination.

### **4. User Acceptance Tests (UAT)**
*(Define tests from the perspective of the end-user. These should be framed as user stories.)*

*   **UAT Case 1: Core Functionality**
    *   **User Story:** "As a user, I want to receive a daily market briefing so that I can stay informed of key events."
    *   **Acceptance Criteria:**
        *   The briefing is delivered by 08:00 AM local time.
        *   The briefing contains a summary of the top 3 market events.
        *   The sentiment analysis for each event is plausible (Positive, Negative, Neutral).
*   **UAT Case 2: Handling of No News**
    *   **User Story:** "As a user, if there are no significant market events, I want to be notified so that I know the system is still working."
    *   **Acceptance Criteria:**
        *   The system delivers a message like "No significant market-moving events were identified for today's briefing."

### **5. Tool & Dependency Tests**
*(Define tests to ensure the system's external tools are working as expected.)*

*   **Test Case TD-1: Web Search Tool**
    *   **Action:** Have an agent perform a search for a known topic.
    *   **Assertion:** Verify that the tool returns a list of relevant URLs.
*   **Test Case TD-2: API Failure**
    *   **Setup:** Mock the `[External_API]` to return an error (e.g., a 503 status code).
    *   **Action:** Trigger a workflow that depends on the API.
    *   **Assertion:** Verify that the system fails gracefully and logs the error, rather than crashing.

---
```
