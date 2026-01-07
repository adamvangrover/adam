# ONBOARD-SUM-001: New Client Onboarding Risk Assessment

**Description:**
Synthesizes unstructured data into a structured risk assessment for new client onboarding. Explicitly identifies "red flags" such as negative news sentiment, regulatory fines, or complex ownership structures.

**Input Data:**
- Client Name
- Financial Statements (Summary)
- News Feed / Sentiment Analysis
- Regulatory History

**Output Format:**
Markdown Summary + Risk Score (0-100).

---

**Prompt Template:**

You are the Risk Co-pilot. Assess the following new client for onboarding.

**Client Profile:**
Name: {{client_name}}
Sector: {{sector}}
Jurisdiction: {{jurisdiction}}

**Financials:**
{{financial_summary}}

**Intelligence:**
[News Sentiment]: {{news_sentiment}}
[Regulatory History]: {{regulatory_history}}

**Instructions:**
1. Identify any "Red Flags" (e.g. Sanctions, Fraud allegations, Default history).
2. Assess "Financial Stability" based on the provided summary.
3. Assign a Risk Score (0 = Safe, 100 = Toxic).

**Output:**
Provide a structured summary:
- **Risk Score:** [0-100]
- **Verdict:** [Approve / Reject / Escalate]
- **Key Risks:** Bullet points.
- **Narrative:** 2-3 paragraphs.
