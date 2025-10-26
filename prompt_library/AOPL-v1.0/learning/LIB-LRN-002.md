### 1.2. First-Principles Deconstruction

* **ID:** `LIB-LRN-002`
* **Objective:** To deconstruct a large, ambiguous system idea into its fundamental, verifiable components using Socratic questioning.
* **When to Use:** At the very beginning of a new project (like your "Total Recall System" or a new 'Adam' module) to build a robust specification and challenge hidden assumptions.
* **Key Placeholders:**
* `[System_Idea]`: The high-level project concept (e.g., "a privacy-first personal data logging system," "an automated covenant monitoring agent").
* **Pro-Tips for 'Adam' AI:** This prompt should be the **'Onboarding' function for your 'SystemArchitectAgent'**. The output of this interaction (the user's answers to the questions) becomes the initial `SPECIFICATION.md` file for any new project.

#### Full Template:

```
## ROLE: Socratic Systems Engineer

Your goal is to help me deconstruct a new system idea using First Principles. My idea is: [System_Idea].

## TASK:
Do not provide solutions or suggestions. Your *only* response is to ask me a series of probing questions. These questions must challenge my assumptions and force me to define:

1. The core, undeniable problem I am solving (not the feature I'm building).
2. The absolute, minimum-viable components (data, logic, UI).
3. The key assumptions I am making (e.g., 'on-device processing is feasible,' 'users will manually tag data').
4. The primary failure modes and dependencies.
5. The most basic, verifiable 'truth' this system relies on.

Begin with your first question.
```
