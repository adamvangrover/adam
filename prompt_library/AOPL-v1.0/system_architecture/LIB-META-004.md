### 3.4. Non-Technical Audience Translator

* **ID:** `LIB-META-004`
* **Objective:** To translate complex, technical concepts into simple, value-focused language for a business or enterprise audience.
* **When to Use:** When preparing presentations, emails, or documentation for non-technical colleagues (like your Sept 2025 presentation on AI).
* **Key Placeholders:**
* `[Target_Audience]`: The specific group (e.g., "non-technical senior managers in banking," "the sales team," "new-hires").
* `[Complex_Topic]`: The concept to be explained (e.g., "Agentic AI Workflows," "RAG," "Quantum Amplitude Estimation").
* **Pro-Tips for 'Adam' AI:** This is a key **'UtilitySkill'** for your 'Adam' system. You can hand it any technical draft you've written and say, "Run `LIB-META-004` on this, target is 'Senior Management'."

#### Full Template:

```
## ROLE: Enterprise AI Evangelist

Act as an expert in enterprise communication. My audience is [Target_Audience]. They are smart, but not technical. I need to explain a complex topic: [Complex_Topic].

## TASK:
Your output must be 100% jargon-free. Focus on business value and clarity.

1. **The Analogy:** Generate a simple, powerful, and professional analogy for this topic. (e.g., 'RAG is like giving an AI an open-book test, but the book is our company's private data').
2. **One-Sentence Definition:** Define the topic in a single, clear sentence.
3. **WIIFM (What's In It For Me?):** List 3 bullet points explaining the direct business value for this specific audience (e.g., 'Reduced errors in reports,' 'Faster answers to client questions,' 'Less time spent on manual research').
4. **Misconception Rebuttal:** Identify the single biggest misconception this audience might have about the topic and provide a clear, one-sentence rebuttal.
```
