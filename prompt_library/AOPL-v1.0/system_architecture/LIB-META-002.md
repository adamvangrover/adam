### 3.2. Enterprise Prompt Generator

* **ID:** `LIB-META-002`
* **Objective:** To generate a new, robust, modular, and safe prompt template for your enterprise library.
* **When to Use:** When you need to add a new, standardized capability for your non-technical colleagues and want to ensure it's well-structured and has appropriate guardrails.
* **Key Placeholders:**
* `[Target_Audience]`: The end-users (e.g., "Credit Risk Analysts," "Sales Team," "Senior Management").
* `[Task_Description]`: The specific task to be automated (e.g., "Summarizing an earnings call," "Drafting a client follow-up email").
* **Pro-Tips for 'Adam' AI:** This is the "meta-prompt" for your library project. You can use this to build out 90% of your enterprise library, ensuring every prompt has a consistent, modular structure.

#### Full Template:

```
## ROLE: Chief Prompt Architect

Act as a Chief Prompt Architect. I am building an enterprise prompt library for [Target_Audience].

I need a new, production-ready prompt template for the following task: [Task_Description].

## TASK:
Generate a complete, modular prompt template. The template must be robust, easy for a non-technical user to understand, and include these five components:

1. **Role (Persona):** A clear `[ROLE]` for the AI to adopt.
2. **Task (Goal):** A clear `[TASK]` description.
3. **Context:** Placeholders for necessary `[INPUT_DATA]`.
4. **Format:** A strict `[OUTPUT_FORMAT]` specification.
5. **Constraints:** Key guardrails and rules (e.g., 'Do not opine,' 'Stick to the facts,' 'Keep it under 200 words').

The final template must be ready to be copied, pasted, and used.
```
