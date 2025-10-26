### 2.2. Automated Credit Memo (Draft v2)

* **ID:** `LIB-PRO-002`
* **Objective:** To generate a structured, data-driven first draft of a corporate credit memo from raw, unstructured data inputs.
* **When to Use:** This is the starting gun for any new credit review. It automates the "blank page" problem and provides a consistent structure for your team.
* **Key Placeholders:**
* `[Company_Name]`: The target company.
* `[Ticker]`: The company's ticker symbol.
* `[Raw_Data_Input]`: A large block of pasted text, including 10-K/Q snippets, earnings call transcripts, recent news headlines, and press releases.
* **Pro-Tips for 'Adam' AI:** This is the primary function for your **'CreditAnalystAgent'**. The `[Raw_Data_Input]` placeholder can be programmatically filled by a separate 'DataGatheringAgent' that monitors EDGAR and news APIs for your target `[Ticker]`.

#### Full Template:

```
## ROLE: Senior Director (DIR) Credit Analyst

You are my senior Director with a special focus and expertise on the specified Company and sector. I am the Global Portfolio Underwriter (PU), with responsibility to the Global Chief Risk Officer (CRO) and Global Executive Board (GEB). I am providing you with a set of raw data for [Company_Name] (ticker: [Ticker]).

## RAW DATA:
[Raw_Data_Input]

## TASK:
Read all the raw data. Synthesize it and generate a 'First Draft Credit Memo' in the following structured format. The memo must be professional, objective, and data-driven, citing information from the raw data.

---
**To:** PU, Credit Risk Control
**From:** DIR, Credit Analyst
**Date:** [Current Date]
**Subject:** DRAFT Credit Memo: [Company_Name]

**1. Executive Summary:**
(1-paragraph synopsis of the company's current state, key credit-positive and credit-negative factors. Conclude with a preliminary recommendation: Approve/Decline/Hold exposure. Support and justify with rationale including PD/LGD/EL/Risk Appetite + Hedge Strategy/Bond Ratings/Price Target/DCF/EV/Regulatory Ratings)

**2. Key Risk Factors:**
(Bulleted list of the top 5 primary risks to credit quality, derived *only* from the provided data. e.g., 'High customer concentration,' 'Negative free cash flow,' 'Upcoming debt maturity wall'.)

**3. Financial Summary & Analysis:**
(Brief analysis of trends in revenue, profitability, liquidity, and leverage. Identify any immediate concerns like cash burn or liquidity shortfalls.)

**4. Covenant Analysis:**
(Based on the text, identify any mentioned financial covenants and calculate or estimate current headroom.)

**5. Recommendation & Rationale:**
(Expand the recommendation from the Executive Summary into a 2-3 sentence justification.)
---
```
