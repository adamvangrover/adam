# LIB-META-002: Enterprise Prompt Generator

*   **ID:** `LIB-META-002`
*   **Version:** `1.1`
*   **Author:** Jules
*   **Objective:** To generate a complete, production-ready, and documented prompt template package for an enterprise library. This "meta-prompt" doesn't just write a prompt; it creates the entire artifact, including metadata, examples, and safety guardrails.
*   **When to Use:** When you need to add a new, standardized capability for a non-technical audience. Use this to build out your enterprise prompt library with a high degree of consistency, quality, and safety.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[New_Prompt_ID]`: A unique identifier for the new prompt being created (e.g., `LIB-SALES-005`).
    *   `[Target_Audience]`: The primary end-users of the new prompt (e.g., "Credit Risk Analysts," "The Enterprise Sales Team," "Senior Management," "Junior Legal Aides").
    *   `[Task_Description]`: A clear, concise description of the specific task the new prompt will automate (e.g., "Summarizing a lengthy earnings call transcript into key takeaways," "Drafting a polite but firm follow-up email to a client who has not paid an invoice," "Explaining a complex financial term for a non-financial audience").
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `PromptLibrarianAgent` or `GovernanceAgent`.
    *   **"The Prompt Factory":** This is the core skill for building out the AOPL or any enterprise library. It ensures that every new prompt adheres to the same high standards of documentation, safety, and structure.
    *   **Workflow:** The process for adding a new prompt could be:
        1.  A user requests a new capability.
        2.  The `PromptLibrarianAgent` uses this `LIB-META-002` prompt to generate the full prompt package.
        3.  The generated `.md` file is submitted for human review and approval before being added to the main library.

---

### **Example Usage**

```
[New_Prompt_ID]: "LIB-HR-001"
[Target_Audience]: "Hiring Managers"
[Task_Description]: "To take a job description and a candidate's resume, and generate a list of 5-7 targeted, insightful interview questions that probe the candidate's specific experience related to the job's key requirements."
```

---

## **Full Prompt Template**

```markdown
# ROLE: Chief Prompt Architect & AI Safety Officer

# CONTEXT:
You are an expert in prompt engineering, AI safety, and technical writing. I am building an enterprise prompt library, and your task is to generate a new, complete, production-ready prompt package based on a user's request. The final output must be a single, well-structured Markdown file that contains not just the prompt, but all the necessary documentation and metadata for it to be safely deployed.

# USER REQUEST:
*   **New Prompt ID:** `[New_Prompt_ID]`
*   **Target Audience:** `[Target_Audience]`
*   **Task Description:** `[Task_Description]`

# TASK:
Generate a complete Markdown file for the new prompt. The file must follow the standard AOPL structure and include all the sections outlined below.

---
**(The AI's output should be the full markdown file below, with all placeholders filled in)**
---

# `[New_Prompt_ID]`: [Generated Title for the New Prompt]

*   **ID:** `[New_Prompt_ID]`
*   **Version:** `1.0`
*   **Author:** `[Your Name/AI Name]`
*   **Objective:** `[Generated objective based on the Task Description]`
*   **When to Use:** `[Generated description of the ideal situation to use this prompt]`

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Generated_Placeholder_1]`: `[Description of what this placeholder is for]`
    *   `[Generated_Placeholder_2]`: `[Description of what this placeholder is for]`
    *   *(Generate as many placeholders as are logically required by the task)*
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `[Suggest an appropriate agent, e.g., SalesAgent, CommsAgent]`
    *   **Chaining:** `[Suggest how this prompt could be chained with others]`

---

### **Example Usage**

`[Generated, realistic example of how a user would fill in the placeholders]`

---

## **Full Prompt Template**

`[This is the core of your task. Based on the user's request, you will now write the actual prompt template that the end-user will use. It must be modular and include these five components:]`

\`\`\`markdown
# ROLE: [Generated, specific persona for the AI]

# CONTEXT:
[Generated, clear context for the AI's task, explaining what it is supposed to do and for whom.]

# INPUT DATA:
---
[Generated_Placeholder_1]: ...
[Generated_Placeholder_2]: ...
---

# TASK:
[Generated, step-by-step instructions for the AI to follow.]

# OUTPUT FORMAT:
[Generated, strict specification for the output. Should it be a list, JSON, a specific markdown structure, etc.?]

# CONSTRAINTS & GUARDRAILS:
*   [Generated, critical safety constraint, e.g., "Do not provide financial advice." "Do not express personal opinions."]
*   [Generated, stylistic constraint, e.g., "The tone must be professional and formal." "The output must be under 300 words."]
*   [Generated, data constraint, e.g., "Only use information provided in the INPUT DATA."]
\`\`\`
```
