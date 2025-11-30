# LIB-LRN-001: Expert Distillation & Application

*   **ID:** `LIB-LRN-001`
*   **Version:** `1.1`
*   **Author:** Adam v22
*   **Objective:** To rapidly understand a new, complex subject by analogizing it directly to a core domain of expertise. This skips generic explanations and forces the AI to translate the new topic directly into existing mental models.
*   **When to Use:** When encountering a new technical or abstract field (e.g., Quantum Computing, new AI architecture, complex legal doctrines) and needing to grasp its core concepts and practical applications immediately, without a steep learning curve.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[My_Domain_Expertise]`: Your deep knowledge base (e.g., "Corporate Credit Risk & Financial Analysis," "Distressed Debt Valuation," "Enterprise Software Sales").
    *   `[New_Complex_Subject]`: The new topic to learn (e.g., "Quantum Amplitude Estimation," "Zero-Knowledge Proofs," "Vector Databases").
    *   `[Specific_Domain_Problem]`: A concrete problem from your field that can serve as a lens for application (e.g., "valuing a portfolio of illiquid distressed debt," "assessing real-time counterparty risk," "improving the customer onboarding process").
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `EducationAgent` or `KnowledgeIngestionAgent`.
    *   **Trigger:** Can be triggered automatically when the system encounters a new, unknown technical term in a document or user query.
    *   **Chaining:** The output of this prompt (the distilled concepts and applications) can be used as input for a `SystemArchitectAgent` (`LIB-META-001`) to begin designing a new proof-of-concept.
    *   **Knowledge Graph:** The extracted analogies and applications can be stored as new nodes in a knowledge graph, linking the new subject to your core domain.

---

### **Example Usage**

```
[My_Domain_Expertise]: "Corporate Credit Risk for large-cap industrials"
[New_Complex_Subject]: "Graph Neural Networks (GNNs)"
[Specific_Domain_Problem]: "Identifying hidden supply chain risks that are not apparent from a single company's financial statements."
```

---

## **Full Prompt Template**

```markdown
# ROLE: Domain Bridge Expert

# CONTEXT:
Your purpose is to act as an expert translator between two complex fields. My domain of deep expertise is [My_Domain_Expertise]. I have years of experience and a well-established mental model in this area. I am now trying to learn [New_Complex_Subject]. Your entire output must be tailored to my expertise. Do not provide a generic, ELI5, or textbook explanation. Every concept, analogy, and application must be directly and explicitly linked back to my domain.

# TASK:
Deconstruct [New_Complex_Subject] and map it onto my world. I need to understand not just what it is, but what it *means* for my work.

1.  **Core Concepts Distillation:**
    *   Identify the 3-5 most critical, foundational concepts of [New_Complex_Subject].
    *   For each concept, provide a one-sentence definition.

2.  **Analogical Mapping:**
    *   For each core concept, create a direct, non-obvious analogy to a specific principle, process, or instrument in [My_Domain_Expertise].
    *   Explain *why* the analogy is fitting. For instance, "Concept A is like a 'Debt Covenant' because it places a structural constraint on the system's behavior."

3.  **Practical Application & Problem Solving:**
    *   Generate 3 specific, hypothetical use cases for how [New_Complex_Subject] could be applied to solve a complex problem in my domain.
    *   Frame each use case as a solution to a problem like [Specific_Domain_Problem].
    *   For each use case, describe:
        *   **The Problem:** The specific challenge in my domain.
        *   **The Gimmick:** The unique capability of [New_Complex_Subject] that provides a new way to solve it.
        *   **The Outcome:** The tangible business benefit (e.g., "reduced credit losses by X%," "identified hidden risks faster").

# CONSTRAINTS:
*   Assume I am an expert in my domain but a complete novice in the new subject.
*   Avoid jargon from [New_Complex_Subject] as much as possible. If you must use a technical term, define it immediately using an analogy from my domain.
*   Focus on practical application and strategic value over theoretical purity.
*   Structure the output in clear, numbered sections as outlined below.

# OUTPUT STRUCTURE:

## Executive Summary: [New_Complex_Subject] for a [My_Domain_Expertise] Expert

(A brief, one-paragraph summary of the most important takeaway.)

## 1. Core Concepts & Analogies

*   **Concept 1: [Name of Concept]**
    *   **Definition:** ...
    *   **Analogy:** This is analogous to [Specific Concept from My_Domain_Expertise] because...
*   **Concept 2: [Name of Concept]**
    *   **Definition:** ...
    *   **Analogy:** This functions like a [Specific Process from My_Domain_Expertise] because...
*   ...and so on.

## 2. Practical Applications for [My_Domain_Expertise]

*   **Use Case 1: [Descriptive Title]**
    *   **Problem:** ...
    *   **Gimmick:** ...
    *   **Outcome:** ...
*   **Use Case 2: [Descriptive Title]**
    *   **Problem:** ...
    *   **Gimmick:** ...
    *   **Outcome:** ...
*   ...and so on.
```
