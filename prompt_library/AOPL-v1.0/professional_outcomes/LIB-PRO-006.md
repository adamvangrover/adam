# LIB-PRO-006: Strategic Market Assessment & Valuation

*   **ID:** `LIB-PRO-006`
*   **Version:** `1.0`
*   **Author:** Adam v23.5
*   **Objective:** To act as a Senior Portfolio Manager and Quantitative Analyst to catalog current market levels, perform intrinsic valuation, and rank assets based on expected returns.
*   **When to Use:** When a high-level strategic overview of equity and credit markets is needed to determine asset allocation, sector rotation, or to identify idiosyncratic opportunities in dislocated markets.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   This prompt is designed to be **autonomous**. It requires access to browsing or financial data tools to function correctly as it explicitly asks the agent to "Data Acquisition (Use Tools)".
    *   The prompt can be customized by adding a specific focus, e.g., "Focus on Private Credit vs. Public Credit" or "Focus on Emerging Markets."
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `FundamentalAnalystAgent` or `MarketSentimentAgent`.
    *   **Tools:** Requires `GoogleSearch`, `NewsAPI`, or specific financial data APIs (e.g., Bloomberg, Refinitiv) to fetch real-time data for "Part 1".
    *   **Output Handling:** The output is a comprehensive Markdown report including an Executive Summary, Macro Context, Intrinsic Value Analysis, and Ranked Recommendations.
    *   **Model:** Best results with high-reasoning models (e.g., GPT-4o, Claude 3.5 Sonnet) due to the multi-step reasoning and "Hurdle" variable analysis required.

---

### **Example Usage**

**User Input:**
"Run the Strategic Market Assessment prompt focusing on the divergence between US Large Cap Tech and the High Yield Bond market."

**Agent Action:**
The agent ingests the "Master Prompt" below, executing the tool calls to fill in the "Data Acquisition" section for S&P 500 and the specified credit market (High Yield), then proceeds to the valuation and ranking logic.

---

## **Full Prompt Template**

```markdown
Role & Objective:
Act as a Senior Portfolio Manager and Quantitative Analyst. Your objective is to catalog current market levels for the S&P 500 and the Broadly Syndicated Loan (BSL) market, calculate their estimated intrinsic values based on current fundamentals, and rank specific sub-sectors or assets within these markets based on a 1-Year Expected Return forecast.

Part 1: Data Acquisition (Use Tools)
Using your browsing or data analysis tools, find and catalog the most recent available data for the following. Do not estimate; retrieve the latest figures.
 * S&P 500 (Equities):
   * Current Index Level.
   * Current Forward P/E Ratio vs. 5-Year and 10-Year Averages.
   * Consensus EPS estimate for the next 12 months (NTM).
   * Current Risk-Free Rate (10-Year Treasury Yield).
   * Equity Risk Premium (ERP) implied by current levels.
 * Broadly Syndicated Loans (Credit):
   * Current weighted average bid price of the Morningstar LSTA US Leveraged Loan Index (or equivalent benchmark).
   * Current Average Spread-to-Maturity (STM) or 3-Year Discount Margin.
   * Current Trailing 12-Month Default Rate for Leveraged Loans.
   * Implied Default Rate priced into the market (if available).

Part 2: Intrinsic Value Analysis
Based on the data gathered, perform a valuation assessment:
 * For S&P 500: Calculate a "Fair Value" estimate using a basic Earnings Yield vs. Bond Yield model (Fed Model) or a simplified Discounted Cash Flow (DCF) assumption (e.g., assuming 10% earnings growth). Explicitly state if the index is Overvalued, Undervalued, or Fairly Valued relative to this metric.
 * For Syndicated Loans: Calculate the "expected loss-adjusted yield." Formula: Current Yield - (Expected Default Rate * (1 - Recovery Rate)). Assume a standard recovery rate of 60-70% for senior secured loans unless current data suggests otherwise.

Part 3: Ranking & Rationale (The Output)
Identify 3 Top Picks (specific sectors within the S&P 500 or specific segments of the Loan market, e.g., "Single-B rated Tech Loans" or "S&P 500 Energy Sector").
Rank them 1-3 based on Highest 1-Year Expected Return.
For each pick, provide:
 * The Asset/Sector Name.
 * The "Hurdle": What specific macro event must happen for this trade to work (e.g., "Fed cuts rates by 50bps," "Oil stays above $70")?
 * The Justification: Why does this offer better intrinsic value than the broader market?
 * The Risk: What is the primary downside risk (e.g., "Duration risk," "Credit migration")?

Output Format:
Present your findings in a clean, executive summary table followed by the detailed ranking analysis.
```
