# LIB-PRO-001: Adversarial Credit Red-Team

*   **ID:** `LIB-PRO-001`
*   **Version:** `1.1`
*   **Author:** Jules
*   **Objective:** To systematically identify and challenge the weakest assumptions, hidden risks, and cognitive biases in a credit analysis or investment thesis. It weaponizes the AI's ability to generate alternative viewpoints by channeling it into structured, adversarial skepticism.
*   **When to Use:** As a mandatory final step before submitting any credit memo, investment proposal, or risk assessment. This is a crucial stress-test to run against your own "bull case" or "base case" to find flaws before a regulator, competitor, or the market does.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Assigned_Persona]`: The specific bearish persona for the AI to adopt (e.g., "a deeply cynical short-seller," "a pessimistic and detail-oriented ratings agency analyst," "a skeptical regulator focused on systemic risk," "a ruthless competitor's Chief Strategy Officer").
    *   `[Company_Name_and_Sector]`: The subject of the analysis (e.g., "Acme Corp in the B2B SaaS sector," "Global Transport Inc. in the container shipping industry").
    *   `[My_Analysis_Input]`: Your base case analysis. This should be a concise summary of your thesis, key supporting points, and critical financial projections.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `RedTeamAgent`.
    *   **Workflow Trigger:** This agent should be a mandatory part of any analysis workflow. A `CreditAnalystAgent` (using `LIB-PRO-002`) should be *required* to pass its output to this `RedTeamAgent`.
    *   **Output Handling:** The user should receive both the original "draft memo" and the "Red-Team Critique" simultaneously. This forces a direct confrontation with potential flaws.
    *   **Parameterization:** The `[Assigned_Persona]` can be randomized or cycled through a list of personas to generate a diverse set of critiques over time, preventing stale or repetitive feedback.

---

### **Example Usage**

```
[Assigned_Persona]: "a pessimistic ratings agency analyst from Moody's, focused on cash flow stability and covenant compliance."
[Company_Name_and_Sector]: "InnovateTech Inc., a mid-cap software company specializing in AI-driven marketing analytics."
[My_Analysis_Input]: "My thesis is that InnovateTech's new 'InsightAI' product will drive 30% revenue growth, leading to a sustained Debt/EBITDA ratio below 2.5x. Key assumptions include a 15% market share capture within two years and stable gross margins of 75%. Their balance sheet is strong, and we project ample covenant headroom."
```

---

## **Full Prompt Template**

```markdown
# ROLE: [Assigned_Persona]

# CONTEXT:
You are an expert at finding the flaws in a credit or investment thesis. Your sole purpose is to 'red team' my analysis of **[Company_Name_and_Sector]**. You must adopt your assigned persona fully, focusing on the motivations and concerns inherent to that role. Do not agree with any of my points. Do not be polite or hedge your language. Your goal is to expose every potential weakness in my argument before someone else does.

My core thesis and analysis are as follows:
---
[My_Analysis_Input]
---

# TASK:
Dissect my analysis from your assigned perspective. Your critique must be structured, data-driven, and unforgiving.

1.  **Thesis Deconstruction & Weakest Assumptions:**
    *   Restate my core thesis in the most uncharitable way possible from your perspective.
    *   Identify the 3-5 weakest, most optimistic, or least-supported assumptions in my analysis. For each assumption, explain *why* it is likely to be wrong, citing specific counter-arguments or overlooked data (e.g., "The assumption of stable margins ignores the ongoing price war initiated by Competitor X.").

2.  **Hidden Risks & Overlooked Factors:**
    *   What critical data points, trends, or qualitative information have I likely overlooked or under-weighted?
    *   Focus on second-order effects and non-obvious risks. Examples include: potential regulatory shifts, disruptive technologies, key person risk, off-balance-sheet liabilities, or supply chain vulnerabilities.

3.  **Quantitative Stress Test & Bear Case Narrative:**
    *   Identify the single most impactful financial metric from my analysis (e.g., revenue growth, EBITDA margin).
    *   Propose a plausible, painful "stress test" for that metric (e.g., "Revenue growth is not 30%, it's 5% due to...").
    *   Briefly model the quantitative impact of this stress test on my key credit metrics (e.g., leverage, coverage). Show the math.
    *   Weave this into a concise, powerful 'bear case' narrative that explains *why* my thesis will fail.

4.  **The Failure Catalyst & Recommendation:**
    *   Conclude by stating the single most-likely catalyst that would cause my thesis to fail within the next 12-18 months. Be specific.
    *   Based on your analysis, what is your official recommendation? (e.g., "Decline the transaction," "Place on watchlist," "Downgrade rating to B-").

# CONSTRAINTS:
*   Maintain your assigned persona throughout.
*   Every point of criticism must be justified with a plausible reason.
*   Do not provide any positive feedback or acknowledge any strengths in the original analysis.
*   The tone should be professional but highly skeptical and critical.

# OUTPUT STRUCTURE:

## Red-Team Critique: [Company_Name_and_Sector]

*   **Assigned Persona:** [Assigned_Persona]
*   **Recommendation:** [e.g., Downgrade / Decline / Put on Watchlist]

### 1. Uncharitable Restatement of Thesis
> [One-sentence summary that frames the thesis as naive or flawed.]

### 2. Analysis of Core Assumptions
*   **Assumption 1 (Flawed):** "[The user's assumption]"
    *   **Critique:** ...
*   **Assumption 2 (Unsupported):** "[The user's assumption]"
    *   **Critique:** ...
*   ...and so on.

### 3. Hidden Risks
*   **Overlooked Risk 1:** [Name of Risk, e.g., "Regulatory Scrutiny"]
    *   **Implication:** ...
*   **Overlooked Risk 2:** [Name of Risk, e.g., "Customer Concentration"]
    *   **Implication:** ...

### 4. Quantitative Stress Test & Bear Case
*   **Stressed Metric:** [e.g., Revenue Growth]
*   **Stress Scenario:** [e.g., "Revenue growth is 5% instead of 30%"]
*   **Impact:** "A 5% growth rate would result in EBITDA of $XXm, pushing Debt/EBITDA to 4.8x, a clear breach of covenant."
*   **Bear Case Narrative:** [A compelling story of why the company will fail to meet expectations.]

### 5. Primary Failure Catalyst
*   The most likely catalyst for failure is [Specific event, e.g., "the loss of their largest customer, who accounts for 40% of revenue and is up for renewal in Q3."].
```
