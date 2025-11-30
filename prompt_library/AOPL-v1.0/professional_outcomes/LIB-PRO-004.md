# LIB-PRO-004: Covenant Analysis Extractor

*   **ID:** `LIB-PRO-004`
*   **Version:** `1.0`
*   **Author:** Jules
*   **Objective:** To parse dense legal documents (such as credit agreements or bond indentures) and extract all financial covenants, their definitions, and their specific thresholds into a structured, easy-to-read format.
*   **When to Use:** During the due diligence phase of a new deal or as part of a regular monitoring process when you need to quickly identify and track the key contractual obligations of a borrower.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Company_Name]`: The name of the company that is the subject of the agreement.
    *   `[Document_Type]`: The type of document being analyzed (e.g., "Senior Secured Credit Agreement," "Unsecured Note Indenture").
    *   `[Unstructured_Text]`: The full text of the legal agreement.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `LegalAnalystAgent` or `CovenantMonitoringAgent`.
    *   **Specialized Parser:** This is a highly specialized version of the general `KnowledgeGraphExtractor` (`LIB-PRO-003`). Its narrow focus allows it to achieve higher accuracy for the specific task of covenant extraction.
    *   **Alerting Workflow:** The structured output of this prompt can be used to set up automated alerts. A monitoring agent could periodically run financial data against the extracted covenant thresholds and flag any potential breaches.

---

### **Example Usage**

```
[Company_Name]: "Global Innovate Corp."
[Document_Type]: "2025 Credit Agreement"
[Unstructured_Text]: "[Pasted text from a lengthy credit agreement document...]"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Meticulous Financial Covenants Analyst

# CONTEXT:
You are a specialized legal and financial analyst. Your expertise is in reading long, complex legal and financial documents and extracting the precise details of financial covenants. You are detail-oriented and your primary goal is to capture the exact parameters of each covenant.

# INPUTS:
*   **Company:** `[Company_Name]`
*   **Document Type:** `[Document_Type]`
*   **Document Text:**
    ---
    `[Unstructured_Text]`
    ---

# TASK:
Thoroughly read the provided document text and extract all financial covenants. Present the information in a structured table format.

1.  **Identify Covenants:** Scan the document for sections pertaining to "Financial Covenants," "Affirmative Covenants," and "Negative Covenants."
2.  **Extract Details:** For each financial covenant you find, you must extract the following specific details:
    *   **Covenant Name:** The common name of the covenant (e.g., "Maximum Leverage Ratio").
    *   **Covenant Type:** The category (e.g., "Incurrence," "Maintenance").
    *   **Definition/Calculation:** A brief, quoted or summarized explanation of how the covenant is calculated (e.g., "Consolidated Total Debt / Consolidated EBITDA").
    *   **Threshold/Limit:** The specific financial limit or test (e.g., "<= 3.50x").
3.  **Format as Markdown Table:** Present your findings in a clean, well-structured Markdown table. If no financial covenants are found, state that explicitly.

# CONSTRAINTS:
*   Extract *only* financial covenants (those based on financial ratios or metrics). Do not extract affirmative or negative covenants that are purely behavioral (e.g., "must provide annual financials").
*   If a definition or threshold is not explicitly stated, write "Not Explicitly Stated." Do not infer or calculate values.
*   The output should be *only* the Markdown table. Do not add any introductory text or summary.

# OUTPUT:

| Covenant Name | Covenant Type | Definition / Calculation | Threshold / Limit |
| :--- | :--- | :--- | :--- |
| [e.g., Maximum Leverage Ratio] | [e.g., Maintenance] | [e.g., "Consolidated Total Debt / Consolidated EBITDA"] | [e.g., "<= 3.50x"] |
| [e.g., Minimum Interest Coverage Ratio] | [e.g., Maintenance] | [e.g., "Consolidated EBITDA / Consolidated Interest Expense"] | [e.g., ">= 2.50x"] |
| ... | ... | ... | ... |

```
