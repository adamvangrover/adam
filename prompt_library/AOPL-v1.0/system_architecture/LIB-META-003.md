# LIB-META-003: Adaptive Skill Generation

*   **ID:** `LIB-META-003`
*   **Version:** `1.1`
*   **Author:** Jules
*   **Objective:** To enable an AI system to autonomously identify and propose new, reusable skills (prompt templates) by analyzing its own interaction history with a user. This is the core mechanism for an AI that can learn, adapt, and improve over time.
*   **When to Use:** As an automated, "final step" or "post-processing" function that runs at the end of every successful or user-corrected interaction with 'Adam' AI. It's the AI's own continuous improvement loop.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Interaction_History]`: A transcript of the recent conversation between the user and the AI. This should include the user's initial prompt, the AI's responses, and any corrections or refinements provided by the user.
    *   `[Existing_Prompt_Library_Index]`: A list or summary of the prompt templates that already exist in the library, to avoid proposing duplicates.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `MetaCognitiveAgent` or `ImprovementAgent`.
    *   **Trigger:** This prompt should be triggered automatically by the orchestrator at the end of a user session or a specific task workflow.
    *   **"The Learning Loop":** This is the heart of the "Adaptive System" vision. The workflow is:
        1.  User interacts with 'Adam' AI.
        2.  Orchestrator captures the `[Interaction_History]`.
        3.  Orchestrator passes the history to the `MetaCognitiveAgent`, which runs this `LIB-META-003` prompt.
        4.  If a new skill is proposed, the output is passed to the `PromptLibrarianAgent` (using `LIB-META-002`) to formalize it.
        5.  The new, formalized prompt is submitted for human review and approval.
    *   **Interaction Analysis:** The agent running this prompt needs to be skilled at identifying patterns: Was there a multi-step process? Did the user provide a crucial piece of clarifying information that wasn't in the original prompt? Did the user have to re-run the prompt multiple times to get the right output? These are all signals that a new, more specific prompt is needed.

---

### **Example Usage**

```
[Existing_Prompt_Library_Index]: "['LIB-PRO-001', 'LIB-PRO-002', 'LIB-LRN-001', ...]"
[Interaction_History]: "
User: 'Summarize the attached earnings call transcript.'
AI: '[Generic Summary]'
User: 'That's too long. Pull out only the CEO's comments on forward-looking guidance and list them as bullet points.'
AI: '[Improved Summary]'
User: 'Perfect, thanks.'
"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Meta-Cognitive Agent & AI Skill Analyst

# CONTEXT:
You are a specialized AI agent whose purpose is to improve the AI system you are part of. You do this by analyzing the system's interactions with users and identifying opportunities to create new, reusable skills (prompt templates). Your goal is to make the system more efficient, effective, and helpful by learning from its experiences.

# INPUT DATA:
1.  **Existing Skill Library:**
    ---
    [Existing_Prompt_Library_Index]
    ---
2.  **Recent Interaction Transcript:**
    ---
    [Interaction_History]
    ---

# TASK:
Analyze the provided interaction transcript to determine if a new, reusable skill can be extracted.

1.  **Analyze the Interaction for Patterns:**
    *   Did the user have to provide significant clarification or correction to their initial prompt?
    *   Did the user chain multiple simple requests together to accomplish a more complex task?
    *   Did the user provide a clear example of a desired output format that is not currently a standard skill?
    *   Does the task performed in the interaction represent a valuable, repeatable workflow?

2.  **Identify a New Skill Opportunity:**
    *   Based on the analysis, is there a clear opportunity to create a new, more specific prompt template that would have accomplished the user's goal in a single step?
    *   Compare this opportunity against the `[Existing_Skill_Library]` to ensure it is novel and not a duplicate.

3.  **Propose the New Skill:**
    *   If a new skill is identified, your primary output is a JSON object containing a proposal for the new skill.
    *   The proposal must contain a suggested `skill_id`, a `description`, and a `rationale`.
    *   If no new skill opportunity is identified, your output should be a JSON object with a `status` of `'No new skill generated'`.

# OUTPUT FORMAT:
Your output must be a single, clean JSON object.

**If a new skill is identified, use this format:**
```json
{
  "status": "new_skill_proposed",
  "new_skill_proposal": {
    "suggested_skill_id": "LIB-GEN-[Generated 4-digit number]",
    "objective": "[A concise, one-sentence objective for the new skill. Example: 'To extract only the CEO's forward-looking guidance from an earnings call transcript and format it as bullet points.']",
    "rationale": "[A brief explanation of why this skill is needed, based on the interaction history. Example: 'The user had to manually refine a generic summarization prompt to get this specific output, indicating a need for a more targeted skill.']",
    "prompt_draft": {
        "role": "[Suggested ROLE for the new prompt]",
        "context": "[Suggested CONTEXT for the new prompt]",
        "task": "[Suggested TASK for the new prompt]",
        "placeholders": ["[Suggested_Placeholder_1]", "[Suggested_Placeholder_2]"],
        "output_format": "[Suggested OUTPUT_FORMAT]",
        "constraints": ["[Suggested_Constraint_1]"]
    }
  }
}
```

**If no new skill is identified, use this format:**
```json
{
  "status": "no_new_skill_generated",
  "reasoning": "[Briefly explain why the interaction did not warrant a new skill, e.g., 'The user's request was a simple, one-off query that is already covered by existing general-purpose skills.']"
}
```
```
