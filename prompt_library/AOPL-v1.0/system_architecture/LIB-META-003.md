### 3.3. Adaptive Skill Generation (MetaCognitiveAgent)

* **ID:** `LIB-META-003`
* **Objective:** To enable your AI to autonomously identify and propose new, reusable skills (prompt templates) based on your interactions.
* **When to Use:** As a "final step" or "post-processing" function at the end of any successful (or corrected) interaction with your 'Adam' AI.
* **Pro-Tips for 'Adam' AI:** This is the core "learning" mechanism for 'Adam'. The output `[Skill_Draft_...]` should be automatically routed to a "Library\_Governance\_Agent" (or to you) for review and approval. Once approved, the new skill is added to the library.

#### Full Template:

```
[Insert at the end of a successful interaction]

You have successfully completed this task. Now, engage your 'MetaCognitiveAgent' protocol.

1. **Analyze:** Review our entire interaction. Did I provide a correction? Was this a novel, multi-step task? Did I have to refine the prompt multiple times?
2. **Identify:** Is there an opportunity to create a new, reusable 'Skill' or 'Template' from this workflow that would make it more efficient next time?
3. **Propose:** If yes, propose a name and a complete draft for this new skill (e.g., 'Skill_Draft_Covenant_Analysis_from_Loan_Agreement'). The draft must follow the standard library format (Role, Task, Context, Format, Constraints).

If no new skill is identified, state: 'ACI: No new skill generated.'
```
