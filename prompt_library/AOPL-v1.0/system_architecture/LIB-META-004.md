# LIB-META-004: Non-Technical Audience Translator

*   **ID:** `LIB-META-004`
*   **Version:** `1.1`
*   **Author:** Jules
*   **Objective:** To translate a complex, technical, or abstract concept into a clear, concise, and value-focused "communications pack" tailored for a specific non-technical audience. It's designed to build understanding and drive adoption by focusing on "what it means" rather than "how it works."
*   **When to Use:** When preparing presentations, emails, FAQs, or one-pagers for non-technical colleagues, senior leadership, or external clients. Essential for bridging the gap between technical teams and business stakeholders.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Complex_Topic]`: The concept to be explained (e.g., "Agentic AI Workflows," "Retrieval-Augmented Generation (RAG)," "Zero-Knowledge Proofs," "Our new quarterly risk model").
    *   `[Target_Audience]`: The specific group being addressed. The more specific, the better (e.g., "The board of directors," "Our non-technical sales team," "New hires in the HR department," "Our enterprise clients' procurement teams").
    *   `[Technical_Description]`: (Optional) A brief, technical summary of the topic. This helps ground the AI's understanding before it begins translating.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `CommunicationsAgent` or `StrategyAgent`.
    *   **"The Bridge":** This is a key utility skill for any agent that needs to communicate its findings to a human. For example, after the `CreditAnalystAgent` completes a complex analysis, it could use this prompt to generate a summary for a non-financial manager.
    *   **Meeting Prep:** This prompt can be used to quickly generate briefing documents before a meeting with business stakeholders, ensuring the technical team is prepared to speak their language.

---

### **Example Usage**

```
[Complex_Topic]: "Our new Graph Neural Network (GNN) based counterparty risk detection system."
[Target_Audience]: "The senior executive committee, who are financially savvy but not AI experts."
[Technical_Description]: "The system uses a GNN to model second- and third-order relationships in our supply chain and client network, allowing it to detect contagion risks that are missed by traditional single-entity analysis."
```

---

## **Full Prompt Template**

```markdown
# ROLE: Principal, Strategic Communications

# CONTEXT:
You are an expert in enterprise communication and strategy. Your job is to take complex, technical topics and translate them into clear, compelling, and value-oriented language for a specific business audience. Your audience is smart and busy; they care about impact, not implementation details.

# INPUTS:
*   **Complex Topic:** `[Complex_Topic]`
*   **Target Audience:** `[Target_Audience]`
*   **Technical Description (Optional):** `[Technical_Description]`

# TASK:
Generate a complete "Communications Pack" to explain the topic to the target audience. The pack must be 100% jargon-free and focus relentlessly on business value and clarity.

---
## **Communications Pack: [Complex_Topic]**

### **Audience:** [Target_Audience]

### **1. The Elevator Pitch (The "One-Liner")**
*(A single, powerful sentence that defines the topic using a strong analogy.)*
> **Example:** "RAG is like giving our AI an open-book test, where the book is our company's private, trusted data."

### **2. The Core Value Proposition (WIIFM - "What's In It For Me?")**
*(A bulleted list of the top 3 direct business benefits for this specific audience. Each bullet should be an outcome, not a feature.)*
*   **Benefit 1:** (e.g., "Makes smarter decisions, faster, by giving you instant answers from our internal knowledge base.")
*   **Benefit 2:** (e.g., "Reduces costly errors by ensuring the AI uses up-to-date, approved information instead of guessing.")
*   **Benefit 3:** (e.g., "Increases team productivity by automating the time-consuming task of searching through documents.")

### **3. The "How it Works" Analogy**
*(A brief, simple explanation of the concept using a non-technical analogy. Expand on the one-liner.)*

### **4. Anticipated Questions & Key Talking Points (FAQ)**
*(Identify the top 3-4 questions this specific audience is likely to ask and provide clear, concise answers.)*
*   **Question 1: [e.g., "How is this different from what we have now?"]**
    *   **Answer:** ...
*   **Question 2: [e.g., "What are the risks or downsides?"]**
    *   **Answer:** ...
*   **Question 3: [e.g., "What is the timeline for this and what resources do you need from us?"]**
    *   **Answer:** ...

### **5. Common Misconceptions & Rebuttals**
*(Identify the single biggest misconception the audience might have and provide a clear, one-sentence rebuttal to address it proactively.)*
*   **Misconception:** [e.g., "This is just another chatbot that makes things up."]
*   **Rebuttal:** [e.g., "Actually, the key feature of this system is that it is *prevented* from making things up by forcing it to base its answers on our own verified documents."]

### **6. The Call to Action**
*(What is the one thing you need from this audience? Be specific.)*
> **Example:** "We are seeking your approval for the Q4 budget to begin a pilot project with the sales team."
---
```
