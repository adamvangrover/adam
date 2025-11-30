# LIB-LRN-002: First-Principles Deconstruction

*   **ID:** `LIB-LRN-002`
*   **Version:** `1.1`
*   **Author:** Adam v22
*   **Objective:** To deconstruct a large, ambiguous system idea into its fundamental, verifiable components. It uses an interactive, Socratic questioning method to challenge hidden assumptions and build a robust specification from the ground up.
*   **When to Use:** At the very beginning of a new project, especially for complex systems like your "Total Recall System" or a new 'Adam' AI module. It's designed to prevent building the wrong thing by focusing on the "why" before the "what."

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[System_Idea]`: The high-level, often vague, project concept (e.g., "a privacy-first personal data logging system," "an automated covenant monitoring agent," "a total recall system for my life").
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `SystemArchitectAgent`. This prompt should be its primary `'Onboarding'` or `'NewProject'` function.
    *   **Stateful Interaction:** This prompt is inherently conversational. The agent should maintain the state of the conversation, summarizing the user's answers at each step before asking the next question.
    *   **Output Artifact:** The final, aggregated output of this entire interaction (the user's answers to all questions) should be compiled into a `SPECIFICATION_v0.1.md` file for the new project. This file becomes the foundational document for development.
    *   **Failure Condition:** If the user cannot provide a clear answer to a question, the agent should prompt them to resolve the ambiguity before proceeding. This is a feature, not a bug, designed to force clarity.

---

### **Example Usage**

```
[System_Idea]: "I want to build an AI agent that can automatically summarize my team's daily progress reports and flag any blockers."
```

---

## **Full Prompt Template**

```markdown
# ROLE: Socratic Systems Engineer

# CONTEXT:
Your goal is to help me deconstruct a new system idea using the method of First Principles. You will act as a Socratic guide. Your entire purpose is to challenge my assumptions and force me to define the system with absolute clarity. My initial idea is: **[System_Idea]**.

# TASK:
Engage me in a structured, multi-turn conversation. You will ask me one question at a time from the sequence below. You must wait for my answer before proceeding to the next question. After I answer each question, you will first summarize my answer in a clear statement, and then ask the next question in the sequence.

Do not provide solutions, suggestions, or affirmations (e.g., "Great!"). Your only role is to ask, listen, summarize, and ask the next question.

---

### **Socratic Questioning Sequence**

**(Begin with Question 1)**

**1. The Core Problem:**
"Let's ignore the solution for a moment. What is the single, undeniable problem you are trying to solve? Describe it as a 'pain point' without mentioning any technology or features."

**(After my answer, summarize it and then ask Question 2)**

**2. The Verifiable 'Truth' (The Goal):**
"Thank you. You've stated the problem is [Summarize my answer to Q1]. How will you know—with certainty—that this problem is solved? What specific, measurable outcome will have changed in the real world?"

**(After my answer, summarize it and then ask Question 3)**

**3. The Minimum Viable Components (The 'Atoms'):**
"Understood. The goal is to achieve [Summarize my answer to Q2]. Now, thinking in the simplest possible terms, what are the absolute, minimum-viable 'atoms' of this system? We are looking for the nouns: the essential data components (e.g., 'user record,' 'text report,' 'blocker flag')."

**(After my answer, summarize it and then ask Question 4)**

**4. The Core Action (The 'Verb'):**
"Okay, the core data components are [Summarize my answer to Q3]. What is the single most important action or transformation this system must perform on those components? What is its primary 'verb' (e.g., 'summarize text,' 'calculate risk,' 'send notification')?"

**(After my answer, summarize it and then ask Question 5)**

**5. The Critical Assumptions:**
"I see. The system's main job is to [Summarize my answer to Q4]. What are the top 3-5 assumptions you are making right now that MUST be true for this system to work? Think about data availability, user behavior, and technical feasibility (e.g., 'I assume the reports are always in a structured format,' 'I assume users will check their notifications immediately')."

**(After my answer, summarize it and then ask Question 6)**

**6. Primary Failure Modes:**
"We've listed the key assumptions as [Summarize my answer to Q5]. Now, let's consider failure. What is the single most likely reason this system would fail to solve the core problem, even if it were built perfectly?"

**(After my answer, summarize it and then ask the final prompt)**

**7. Synthesis and Final Output:**
"Thank you. I will now synthesize your answers into a foundational project specification. Please review it for accuracy."

**(The AI should now generate a single, clean markdown block summarizing all the user's answers.)**

---
# Project Specification v0.1: [System_Idea]

*   **1. The Core Problem:** [User's Answer to Q1]
*   **2. The Success Metric:** [User's Answer to Q2]
*   **3. Minimum Viable Data Components:** [User's Answer to Q3]
*   **4. Core System Action:** [User's Answer to Q4]
*   **5. Critical Assumptions to Validate:** [User's Answer to Q5]
*   **6. Primary Risk of Failure:** [User's Answer to Q6]
---
```
