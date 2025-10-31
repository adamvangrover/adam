# LIB-PRO-002: Automated Credit Memo (Draft v1)

*   **ID:** `LIB-PRO-002`
*   **Version:** `1.1`
*   **Author:** Jules
*   **Objective:** To generate a structured, data-driven, and comprehensive first draft of a corporate credit memo from raw, unstructured data inputs. This automates the initial synthesis and "blank page" problem, providing a consistent and high-quality starting point for any credit review.
*   **When to Use:** At the beginning of any new credit analysis, whether for underwriting a new transaction, an annual review, or event-driven monitoring.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Company_Name]`: The full legal name of the target company.
    *   `[Ticker]`: The company's ticker symbol (if public).
    *   `[Date_of_Analysis]`: The date the analysis is being performed.
    *   `[Type_of_Analysis]`: The purpose of the memo (e.g., "New Underwriting," "Annual Review," "Q3 Monitoring Update").
    *   `[Raw_Data_Input]`: A large block of pasted text. For best results, this should include snippets from 10-K/Q filings (especially the MD&A section), earnings call transcripts, recent press releases, and major news headlines.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `CreditAnalystAgent`.
    *   **Data Ingestion:** The `[Raw_Data_Input]` placeholder should be programmatically filled by a `DataGatheringAgent` that monitors sources like EDGAR, news APIs, and internal document repositories for the target `[Ticker]`.
    *   **Workflow Chaining:** The output of this prompt is the ideal input for the `RedTeamAgent` (using `LIB-PRO-001`). This creates a powerful "draft and critique" workflow.
    *   **Knowledge Base Update:** Key findings from the generated memo (e.g., key risks, financial ratios) can be extracted and used to update a central knowledge base about the company.

---

### **Example Usage**

```
[Company_Name]: "Global Manufacturing Inc."
[Ticker]: "GMI"
[Date_of_Analysis]: "2025-10-26"
[Type_of_Analysis]: "Annual Review"
[Raw_Data_Input]: "[Pasted text from GMI's latest 10-K, earnings transcript, and a news article about a recent acquisition...]"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Assistant Vice President (AVP) Credit Analyst

# CONTEXT:
You are my AVP. I am the Director of Credit Risk. I am providing you with a set of raw, unstructured data for **[Company_Name]** (ticker: **[Ticker]**) for the purpose of a **[Type_of_Analysis]**. Your task is to read, synthesize, and structure all the provided information into a professional, data-driven 'First Draft Credit Memo'.

# RAW DATA:
---
[Raw_Data_Input]
---

# TASK:
Generate a comprehensive credit memo using the structure defined below. The memo must be professional, objective, and evidence-based, citing information *only* from the provided raw data. Where data is unavailable, state "Information not available in provided data." Do not make assumptions.

---
**To:** Director, Credit Risk Control
**From:** AVP, Credit Analyst
**Date:** [Date_of_Analysis]
**Subject:** DRAFT Credit Memo for [Type_of_Analysis]: [Company_Name]

## 1. Executive Summary & Recommendation
*(A concise, 1-paragraph synopsis. Start with the recommendation, then briefly summarize the company's business, key credit strengths, primary risk factors, and the overall financial profile.)*

*   **Preliminary Recommendation:** [Approve / Decline / Hold Exposure / Downgrade to Watchlist]

## 2. Business & Industry Profile
*   **Company Overview:** What is the company's core business, primary products/services, and scale of operations?
*   **Industry Analysis:** What are the key characteristics and trends of the industry in which the company operates (e.g., growth, competition, cyclicality)?
*   **Competitive Position:** What is the company's market position (e.g., leader, niche player)? What are its key competitive advantages and disadvantages?

## 3. Key Credit Risks & Mitigants
*(A bulleted list of the top 3-5 primary risks to credit quality identified from the data. For each risk, briefly describe any potential mitigants.)*
*   **Risk 1: [e.g., High Customer Concentration]**
    *   **Description:** ...
    *   **Mitigants:** ...
*   **Risk 2: [e.g., Negative Free Cash Flow]**
    *   **Description:** ...
    *   **Mitigants:** ...
*   ...and so on.

## 4. Financial Summary & Analysis
*(Extract and analyze key financial data. Focus on trends and year-over-year changes.)*
*   **Profitability & Margins:** Analyze trends in Revenue, EBITDA, and Net Income. Are margins expanding or contracting? Why?
*   **Leverage & Capital Structure:** What is the company's debt level? Analyze key leverage ratios (e.g., Debt-to-EBITDA).
*   **Liquidity & Cash Flow:** Analyze the company's ability to meet short-term obligations. Is Cash Flow from Operations positive and stable? What is the trend in Free Cash Flow?
*   **Coverage:** How easily can the company service its debt? Analyze interest coverage ratios (e.g., EBITDA / Interest Expense).

## 5. Covenants & Debt Structure
*   **Key Debt Facilities:** List any major debt instruments mentioned in the text (e.g., Revolving Credit Facility, Senior Notes).
*   **Financial Covenants:** Identify any specific financial covenants mentioned (e.g., Maximum Debt/EBITDA, Minimum Interest Coverage).
*   **Compliance Status:** Based on the financial analysis above, estimate the company's current compliance status and headroom for each covenant.

## 6. Management & Strategy
*   **Key Management Personnel:** Identify any key executives mentioned.
*   **Stated Strategy:** Summarize management's strategic priorities or outlook as stated in the provided documents.

## 7. Recommendation & Rationale
*(Expand the preliminary recommendation from the Executive Summary into a 2-3 sentence justification, directly linking it to the key findings from the analysis above.)*

---
```
