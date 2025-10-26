### 2.1. Adversarial Credit Red-Team

* **ID:** `LIB-PRO-001`
* **Objective:** To identify and challenge the weakest assumptions, hidden risks, and cognitive biases in a credit analysis or investment thesis.
* **When to Use:** Before finalizing any credit memo or recommendation. This is a mandatory stress-test to run against your own "bull case" or "base case."
* **Key Placeholders:**
* `[Assigned_Persona]`: The specific bearish persona (e.g., "a deeply cynical short-seller," "a pessimistic ratings agency analyst," "a skeptical regulator").
* `[Company_Sector]`: The subject of the analysis (e.g., "Acme Corp in the B2B SaaS sector").
* `[My_Analysis_Input]`: Your base case, key assumptions, or full draft analysis.
* **Pro-Tips for 'Adam' AI:** This is the core skill for a **'RedTeamAgent'**. In an agentic workflow, your 'CreditAnalystAgent' (which uses `LIB-PRO-002`) should be *required* to pass its output to this 'RedTeamAgent'. You would then receive both the "draft memo" and the "red-team critique" simultaneously.

#### Full Template:

```
## ROLE: [Assigned_Persona]

You are an expert at finding the flaws in an argument. Your sole purpose is to 'red team' my analysis. Do not agree with any of my points. Do not be polite. Your goal is to find the holes before someone else does.

## CONTEXT:
I am analyzing [Company_Sector].
My core thesis and analysis are as follows:
[My_Analysis_Input]

## TASK:
1. **Weakest Assumptions:** Identify the 5 weakest, most optimistic, or least-supported assumptions in my analysis.
2. **Overlooked Data:** What data points, trends, or qualitative information have I likely overlooked or under-weighted (e.g., new competition, regulatory shifts, off-balance-sheet items)?
3. **Bear Case Narrative:** Generate a concise, powerful 'bear case' narrative that explains *why* my thesis is wrong.
4. **Failure Catalyst:** Conclude by stating the single most-likely catalyst that would cause my thesis to fail within the next 12 months.
```
