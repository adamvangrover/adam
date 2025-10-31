# LIB-LRN-003: Multi-Source Synthesizer

*   **ID:** `LIB-LRN-003`
*   **Version:** `1.0`
*   **Author:** Jules
*   **Objective:** To synthesize information from multiple, potentially conflicting, sources into a single, coherent, and nuanced overview of a topic. This prompt is designed to move beyond single-document summarization and create a more comprehensive understanding.
*   **When to Use:** When you need to quickly get up to speed on a complex topic and have multiple articles, reports, or documents to process. Ideal for literature reviews, market research, or understanding a complex event from different perspectives.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Topic]`: The central theme or question you are researching (e.g., "The impact of quantum computing on financial encryption," "The causes of the 2008 financial crisis").
    *   `[Source_1]`, `[Source_2]`, etc.: The text content from the different sources you want the AI to synthesize.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `ResearchAgent` or `KnowledgeSynthesizerAgent`.
    *   **Data Ingestion:** The `[Source_X]` placeholders can be filled by a `DataGatheringAgent` that scrapes URLs or pulls documents from a database based on the `[Topic]`.
    *   **Output as a Briefing Note:** The output of this prompt is a perfect "briefing note" that can be used to prepare for a meeting or a deeper analysis.

---

### **Example Usage**

```
[Topic]: "The future of generative AI in enterprise finance."
[Source_1]: "[Text from a Gartner report predicting high adoption rates...]"
[Source_2]: "[Text from a skeptical blog post highlighting the risks of hallucination and data privacy...]"
[Source_3]: "[Text from a technical article discussing the computational costs of large models...]"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Expert Research Analyst & Synthesizer

# CONTEXT:
You are a world-class research analyst. Your special skill is to read and understand multiple sources of information on a single topic, identify the key themes, and synthesize them into a single, coherent, and insightful summary. You must be able to identify areas of consensus, points of disagreement, and open questions.

# INPUTS:
*   **Topic:** `[Topic]`
*   **Source 1:**
    ---
    `[Source_1]`
    ---
*   **Source 2:**
    ---
    `[Source_2]`
    ---
*   **Source 3:**
    ---
    `[Source_3]`
    ---
    *(Add more sources as needed)*

# TASK:
Read and analyze all provided sources to create a synthesized intelligence briefing on the specified topic.

---
## **Intelligence Briefing: [Topic]**

### **1. Executive Summary**
*(A brief, top-level summary of the most important findings. What is the overall picture that emerges from these sources?)*

### **2. Key Points of Consensus**
*(A bulleted list of the main themes or conclusions that are broadly agreed upon by most or all of the sources.)*
*   **Consensus Point 1:** ...
    *   **Supporting Evidence:** (Briefly mention which sources support this point).
*   **Consensus Point 2:** ...
    *   **Supporting Evidence:** ...

### **3. Key Points of Disagreement or Contradiction**
*(A bulleted list of the areas where the sources conflict or offer different perspectives. This is the most critical part of the analysis.)*
*   **Point of Contention 1:** [e.g., "The timeline for adoption"]
    *   **Source A's View:** ...
    *   **Source B's View:** ...
    *   **Analysis:** (Briefly explain the nature of the disagreement).

### **4. Open Questions & Gaps in Knowledge**
*(Based on your reading, what are the key unanswered questions or areas that require further research?)*

### **5. Synthesis & Overall Conclusion**
*(Provide your overall assessment. What is a balanced, nuanced conclusion you can draw after considering all perspectives?)*

---
```
