### 3.1. Agentic Framework Architect

* **ID:** `LIB-META-001`
* **Objective:** To design a multi-agent AI system (e.g., AutoGen, CrewAI) to solve a complex, multi-step task.
* **When to Use:** When scoping a new, complex workflow for your 'Adam' AI (e.g., "continuous risk monitoring") and you need to design the interacting agents.
* **Key Placeholders:**
* `[Complex_Task]`: The high-level goal (e.g., "perform continuous risk monitoring for a portfolio of 50 corporate names," "fully automate the quarterly credit review update process").
* `[Agent_Framework_Preference]`: (Optional) A specific framework you are using (e.g., "AutoGen," "CrewAI").
* **Pro-Tips for 'Adam' AI:** This is the "architect of the architects." You use this template to *design* the agentic systems that will then *use* the other templates in this library.

#### Full Template:

```
## ROLE: Expert in AI Agentic Architecture

Act as an expert systems architect specializing in multi-agent frameworks like [Agent_Framework_Preference]. I want to design a multi-agent system to solve a complex task.

## TASK:
[Complex_Task]

## PROPOSED ARCHITECTURE:
Propose a complete system design. Your output must include:
1. **Agents:** Define the necessary agents by role and expertise (e.g., 'Monitoring_Agent', 'Data_Analyst_Agent', 'Report_Summarizer_Agent', 'Red_Team_Agent').
2. **Workflow:** Define the process (e.g., sequential, hierarchical, graph-based). Who is the orchestrator? What are the entry and exit points?
3. **Tools & Skills:** What specific tools (e.g., 'google_search', 'api_financial_data') or skills (e.g., 'LIB-PRO-002', 'LIB-PRO-001') does each agent need access to?
4. **Communication:** How do agents pass information, state, and artifacts? (e.g., 'Summarizer_Agent reads the output from Data_Analyst_Agent').
5. **Final Output:** What is the final, assembled output of the entire system?
```
