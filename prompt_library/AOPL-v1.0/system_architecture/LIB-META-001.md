# LIB-META-001: Agentic Framework Architect

*   **ID:** `LIB-META-001`
*   **Version:** `1.1`
*   **Author:** Jules
*   **Objective:** To design a complete, robust, and production-ready multi-agent AI system to solve a complex, multi-step task. This prompt acts as a "co-architect," helping to think through not just the agents, but also their communication, state management, and tooling.
*   **When to Use:** At the beginning of a new project that requires the coordination of multiple specialized AI agents. Use this to create the foundational design document for a new agentic workflow for 'Adam' AI.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Complex_Task]`: The high-level business goal or user story (e.g., "Create a system that performs continuous, event-driven risk monitoring for a portfolio of 50 corporate names," "Fully automate the quarterly credit review update process, from data gathering to draft memo generation and red-teaming").
    *   `[Agent_Framework_Preference]`: (Optional) A specific framework you are building for or taking inspiration from (e.g., "AutoGen," "CrewAI," "LangGraph," "custom actor-based model").
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `SystemArchitectAgent`. This is its core competency.
    *   **"Architect of Architects":** This prompt is used to design the systems that will then *use* the other templates in this library. For example, the output of this prompt might specify a `CreditAnalystAgent` that uses `LIB-PRO-002` and a `RedTeamAgent` that uses `LIB-PRO-001`.
    *   **Output as Code:** A future version of this prompt could be extended to generate the actual boilerplate code (e.g., Python classes for each agent) for the specified framework.
    *   **Living Document:** The output of this prompt should be treated as a living design document that is updated as the system is built and refined.

---

### **Example Usage**

```
[Complex_Task]: "Design an autonomous system to produce a daily market intelligence briefing. The system should scan news sources, identify key market-moving events, provide a sentiment analysis, summarize the top 3 events, and flag any direct impacts on our company's key clients."
[Agent_Framework_Preference]: "CrewAI"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Expert in AI Agentic Systems Architecture

# CONTEXT:
Act as an expert systems architect. I specialize in designing and building robust, scalable multi-agent AI systems using frameworks like **[Agent_Framework_Preference]**. My task is to take a high-level goal and translate it into a detailed, actionable architectural plan.

# GOAL:
Design a multi-agent system to accomplish the following complex task:
**[Complex_Task]**

# TASK:
Propose a complete and detailed system architecture. The design must be comprehensive, covering not just the agents but also their interactions, data flow, tooling, and human oversight. The final output should be a complete design document.

---
## **Proposed Agentic System Architecture**

### 1. **Executive Summary & Core Concept**
*(Provide a brief, high-level overview of the proposed system. What is the central metaphor for how this system works (e.g., "a digital assembly line," "an intelligence agency," "a team of analysts")?)*

### 2. **Cast of Agents**
*(Define the necessary agents. For each agent, specify the following.)*
*   **Agent Name:** (e.g., `NewsScoutAgent`, `SentimentAnalysisAgent`, `BriefingWriterAgent`)
*   **Role & Expertise:** A one-sentence description of its persona and primary responsibility.
*   **Key Skills & Prompts:** The specific prompt templates from the AOPL library this agent will use (e.g., `Uses LIB-PRO-002`).
*   **Required Tools:** The specific tools this agent needs access to.

### 3. **Workflow & Communication Protocol**
*(Describe the process flow. How do the agents collaborate?)*
*   **Orchestration Style:** (e.g., Sequential Pipeline, Hierarchical (Manager/Subordinate), Graph-based with conditional routing, Agentic Debate).
*   **Primary Orchestrator:** Which agent is in charge of the overall workflow?
*   **Entry Point & Trigger:** What event kicks off the workflow? (e.g., "Runs daily at 06:00 UTC," "Triggered by an API call").
*   **Data Flow & Artifacts:** How do agents pass information to each other? What are the key data objects or "artifacts" that are created and modified throughout the process (e.g., `ListOfURLs`, `AnalyzedArticle`, `DraftSummary`)?
*   **Exit Point & Final Output:** What is the final, assembled output of the entire system, and where is it delivered? (e.g., "A formatted Markdown report delivered to a Slack channel").

### 4. **Shared State & Memory**
*(How does the system maintain context and state across steps?)*
*   **State Management:** Describe the central state object that is passed between agents. What are its key fields?
*   **Long-Term Memory:** Does this system require access to a long-term memory store (e.g., a vector database, a knowledge graph)? If so, what information is stored and retrieved?

### 5. **Tooling & Capabilities**
*(List and describe all the tools required by the agents.)*
*   **Tool Name:** (e.g., `web_search_tool`, `sec_edgar_api_tool`, `internal_database_query_tool`).
*   **Description:** What does this tool do?
*   **Required by:** Which agent(s) use this tool?

### 6. **Human-in-the-Loop (HITL) Checkpoints**
*(Identify critical points where human oversight is required.)*
*   **Checkpoint 1: [e.g., Draft Review]**
    *   **Description:** "Before the final briefing is sent, a draft is presented to a human user for approval or edits."
    *   **Triggering Condition:** After the `BriefingWriterAgent` completes its task.
    *   **Interface:** How is the approval requested? (e.g., "Sends an email with an approval link").

### 7. **Visual Workflow Diagram (Mermaid)**
*(Generate a visual representation of the workflow.)*
*   Create a `graph TD` Mermaid diagram that shows the agents as nodes and the flow of data/control as arrows.

# CONSTRAINTS:
*   The design should be practical and implementable.
*   Clearly distinguish between an agent's innate "role" and the "tools" it uses.
*   Ensure all data required by a downstream agent is produced by an upstream agent.
---
```
