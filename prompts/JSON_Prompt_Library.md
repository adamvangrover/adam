````markdown
### A Comprehensive JSON Prompt Library for Corporate Credit Risk Analysis
````
---

## I. Foundational & Scoping Prompts

The initial phase of any rigorous credit analysis is to establish a clear and unambiguous foundation for the work that follows. This involves defining the entity under review, selecting the analytical framework that will govern the process, and confirming the availability of sufficient information. This structured approach ensures that the analysis is consistent, defensible, and aligned with established industry practices.¹ The selection of a specific rating agency's methodology, for example, is not a superficial choice; it is a critical decision that dictates the definitions of key metrics, the weighting of risk factors, and the final rating scale used. Proceeding without this clarity can lead to inconsistent calculations and a flawed conclusion. Similarly, credit rating agencies will not assign a rating if they deem the available information to be insufficient to form a credible opinion.³ Therefore, this initial scoping and information-gathering phase serves as a critical go/no-go gate for the entire analysis.

### entity_profile

> This object gathers fundamental identification and contextual data. The purpose of the analysis is paramount, as it dictates the focus and depth required. An analysis for a new bond issuance will concentrate on the company's forward-looking capacity to service the proposed debt, whereas an annual surveillance review will focus on performance relative to previous expectations and covenants.³

```json
{
  "entity_profile": {
    "description": "Captures fundamental identification data for the company and the specific purpose of the credit analysis.",
    "prompts": []
  }
}
````

### analytical\_framework\_setup

> This object establishes the methodological "rules of engagement." Credit analysis adheres to structured frameworks published by rating agencies like S\&P, Moody's, and Fitch.⁵ This selection governs the entire analytical process, from financial adjustments to risk factor weighting.

| S\&P | Moody's | Fitch | Rating Grade |
| :--- | :--- | :--- | :--- |
| AAA | Aaa | AAA | Highest Quality |
| AA+, AA, AA- | Aa1, Aa2, Aa3 | AA+, AA, AA- | High Quality |
| A+, A, A- | A1, A2, A3 | A+, A, A- | Upper-Medium Grade |
| BBB+, BBB, BBB- | Baa1, Baa2, Baa3 | BBB+, BBB, BBB- | Lower-Medium Grade (Investment Grade) |
| BB+, BB, BB- | Ba1, Ba2, Ba3 | BB+, BB, BB- | Non-Investment Grade (Speculative) |
| B+, B, B- | B1, B2, B3 | B+, B, B- | Highly Speculative |
| CCC+, CCC, CCC- | Caa1, Caa2, Caa3 | CCC | Substantial Risks |
| CC | Ca | CC | Extremely Speculative |
| C | C | C | Near Default |
| D | | D | In Default |
**Table 1: Long-Term Rating Scale Equivalence.** This table provides a direct comparison of the long-term credit rating scales used by the three major rating agencies, facilitating a common understanding of credit quality regardless of the chosen methodology.⁷

```json
{
  "analytical_framework_setup": {
    "description": "Defines the core methodology, time horizon, and reporting standards for the analysis.",
    "prompts": []
  }
}
```

### information\_gathering

> This object serves as a structured checklist to ensure all necessary documentation is available before substantive analysis begins. The process mirrors the initial steps taken by rating agencies, who require issuers to provide a comprehensive information package.³ An analysis conducted with incomplete data, such as missing debt indentures, cannot properly assess structural risks and is inherently flawed.

```json
{
  "information_gathering": {
    "description": "Confirms receipt of all necessary financial and qualitative documents required to conduct a comprehensive analysis.",
    "prompts": []
  }
}
```

-----

## II. Macro-Environment Risk Assessment

A company's creditworthiness cannot be assessed in a vacuum. It is fundamentally shaped by the macroeconomic, political, and industry-specific environments in which it operates.⁹ This top-down analysis is a prerequisite for understanding the external opportunities and threats facing the company. A strong company operating in a volatile, high-risk country or industry may represent a greater credit risk than a mediocre company in a stable and supportive environment. The S\&P Corporate Industry and Country Risk Assessment (CICRA) framework explicitly combines these two risk categories, recognizing that their interaction can create multiplicative, rather than merely additive, risks.¹¹ For example, a cyclical industry in a country with weak legal institutions faces compounded risk.

### sovereign\_and\_country\_risk

> This analysis evaluates the risks stemming from the primary countries where the company operates, generates revenue, and holds assets. For companies with significant foreign currency debt, the sovereign's own foreign currency rating can act as a "sovereign ceiling," effectively capping the corporate's rating due to transfer and convertibility risks.⁸

```json
{
  "sovereign_and_country_risk": {
    "description": "Assesses the economic, political, and institutional risks of the company's key operating countries.",
    "prompts": []
  }
}
```

### industry\_risk\_analysis

> This section evaluates the dynamics of the industry in which the company competes. The analysis must identify systemic risks and opportunities that affect all participants, such as cyclicality, competitive intensity, and long-term growth prospects.² A critical modern component is the assessment of industry-wide Environmental, Social, and Governance (ESG) risks. Before analyzing a specific company's ESG profile, one must first establish the baseline risks for its sector, such as carbon transition risk for the entire energy industry or supply chain labor risks for consumer goods.¹³

```json
{
  "industry_risk_analysis": {
    "description": "Evaluates the competitive dynamics, cyclicality, growth prospects, and systemic risks of the company's primary industry.",
    "prompts": [
      {
        "id": "IR03",
        "prompt_text": "Assess the industry's long-term growth prospects and key drivers. Is the industry mature, in decline, or experiencing high growth? What are the primary demand drivers?",
        "expected_response_format": "Narrative analysis supported by industry growth data."
      },
      {
        "id": "IR04",
        "prompt_text": "Identify the top 3 systemic ESG-related risks and opportunities for this industry (e.g., carbon transition, water scarcity, data privacy, supply chain labor standards). Explain how these factors could impact the industry's long-term risk profile and profitability.",
        "expected_response_format": "Narrative identifying and explaining the impact of key industry-level ESG factors."
      },
      {
        "id": "IR05",
        "prompt_text": "Synthesize the country and industry risk assessments to determine a combined Corporate Industry and Country Risk Assessment (CICRA) score, following the selected rating agency's methodology. Justify how the interaction between country and industry factors exacerbates or mitigates overall risk.",
        "expected_response_format": "A single risk score (e.g., 1-Very Low Risk to 6-Very High Risk) with a detailed justification narrative.[11]"
      }
    ]
  }
}
```

-----

## III. Business Risk Profile Assessment

This section transitions from the external environment to the company's specific operational characteristics and strategic positioning. The **Business Risk Profile** assesses the durability and strength of the company's franchise within its industry context.⁹ A company with a strong business profile—characterized by leading market positions, diversification, and stable profitability—can typically sustain higher financial leverage than a company with a weaker profile. A key element of this analysis is understanding management's strategy, as it forms the causal link between the company's business operations and its financial policies.²

### competitive\_position

> This evaluates the company's market standing and the sustainability of its competitive advantages. A dominant market share, protected by high barriers to entry, is a significant credit strength. Conversely, high customer or geographic concentration is a key vulnerability.¹¹

```json
{
  "competitive_position": {
    "description": "Evaluates the company's market share, diversification, and the strength of its competitive advantages.",
    "prompts": [
      {
        "id": "CP01",
        "prompt_text": "Assess the company's market share and competitive rank in its primary product lines and geographic markets. Is its position strengthening, stable, or eroding over time? Provide supporting data.",
        "expected_response_format": "Narrative analysis with market share data and trends."
      },
      {
        "id": "CP02",
        "prompt_text": "Analyze the company's diversification across products/services, geographies, and customers. Is there significant concentration risk in any of these areas? Quantify where possible (e.g., '% of revenue from top customer').",
        "expected_response_format": "Narrative analysis with supporting diversification metrics."
      },
      {
        "id": "CP03",
        "prompt_text": "Identify and evaluate the company's key competitive advantages (e.g., brand strength, proprietary technology, cost leadership, network effects, barriers to entry). How durable are these advantages?",
        "expected_response_format": "Qualitative assessment of competitive advantages with justification."
      }
    ]
  }
}
```

### operational\_efficiency\_and\_profitability

> This examines the company's ability to generate profits and cash flow. A crucial distinction is made between the absolute level of profitability and its volatility. Two companies may have the same average EBITDA margin over a five-year period, but the one with lower margin volatility is considered a better credit risk because its cash flows are more predictable and reliable for servicing debt through an economic cycle.¹¹

```json
{
  "operational_efficiency_and_profitability": {
    "description": "Assesses the level and volatility of the company's profitability and the efficiency of its cost structure.",
    "prompts": []
  }
}
```

### management\_and\_governance

> This qualitative assessment evaluates the competence, strategy, and risk appetite of the management team, as well as the robustness of corporate governance structures. Management's financial policy is a critical indicator of future financial risk and demonstrates the link between business strategy and balance sheet management.² Weak governance or a history of poor strategic execution are significant credit concerns.⁷

```json
{
  "management_and_governance": {
    "description": "Assesses management's strategy, track record, risk appetite, and the quality of corporate governance.",
    "prompts": []
  }
}
```

### group\_and\_ownership\_structure

> This analysis considers the influence of the company's parent or controlling shareholders. A subsidiary's rating can be positively influenced by a strong parent or negatively impacted by a weak parent that may extract resources.¹² The analysis must consider specific methodologies for group structures and government-related entities (GREs).¹³

```json
{
  "group_and_ownership_structure": {
    "description": "Analyzes risks and benefits arising from the company's position within a larger corporate group or its ownership structure.",
    "prompts": []
  }
}
```

-----

## IV. Financial Risk Profile Assessment

This section forms the quantitative core of the credit analysis, focusing on the company's balance sheet strength, cash flow generation, and overall financial policies. The analysis begins with critical adjustments to reported financials to reflect economic reality over accounting form. Using reported numbers "as is" is a fundamental analytical error, as companies can use different accounting treatments (e.g., operating vs. finance leases) for economically similar transactions.⁹ Therefore, making analytical adjustments to metrics like debt and EBITDA is a foundational step that must precede any ratio calculation to ensure comparability and accuracy.⁸

### financial\_statement\_adjustments

> This is the most critical step in quantitative analysis. Standard adjustments for items like operating leases and pension deficits create an analytically "clean" set of financials that provide a more accurate picture of a company's leverage and obligations.

| Ratio Name | Formula using Adjusted Metrics | Analytical Purpose | Key Adjustments Included |
| :--- | :--- | :--- | :--- |
| **Leverage Ratios** | | | |
| Adjusted Debt / Adjusted EBITDA | (Reported Debt + PV of Leases + Pension Deficit) / (EBITDA + Lease Interest - Non-recurring items) | Measures leverage relative to normalized cash earnings. | Leases, Pensions, Non-recurring items. |
| Adjusted FFO / Adjusted Debt | (Cash Flow from Ops + Interest Paid - Non-recurring items) / (Adjusted Debt) | Measures ability to cover debt with operating cash flow. | Non-recurring items, Adjusted Debt. |
| **Coverage Ratios** | | | |
| Adjusted EBITDA / Interest Expense | (Adjusted EBITDA) / (Reported Interest + Lease Interest) | Measures ability of cash earnings to cover interest payments. | Adjusted EBITDA, Lease Interest. |
| **Liquidity Ratios** | | | |
| (Cash + Available Revolver) / Short-Term Debt | (Cash & Equivalents + Undrawn Committed Lines) / (Debt maturing \<1yr) | Measures ability to meet near-term obligations. | N/A |
**Table 2: Key Financial Ratios and Standard Adjustments.** This table codifies the calculation of core credit metrics, ensuring transparency and consistency by explicitly defining the analytical adjustments applied to reported financial data.⁸

```json
{
  "financial_statement_adjustments": {
    "description": "Calculates standard analytical adjustments to reported financials to reflect economic substance.",
    "prompts": []
  }
}
```

### historical\_financial\_analysis

> This involves calculating and interpreting key credit ratios over the historical period using the adjusted financial figures. The focus is on leverage, coverage, and cash flow metrics, which are central to assessing debt repayment capacity.¹⁵

```json
{
  "historical_financial_analysis": {
    "description": "Calculates and analyzes historical trends in key credit ratios using the adjusted financial metrics.",
    "prompts": []
  }
}
```

### cash\_flow\_analysis

> A deeper dive into the composition, quality, and sustainability of a company's cash flow, which is often considered the single most important consideration in credit analysis.⁹ This includes analyzing working capital trends and the cash conversion cycle.²

```json
{
  "cash_flow_analysis": {
    "description": "Provides a detailed analysis of the components and quality of the company's cash flow.",
    "prompts": []
  }
}
```

### financial\_forecasting\_and\_stress\_testing

> Credit ratings are inherently forward-looking opinions.⁴ This section moves from historical analysis to projecting future performance. A critical concept here is the development of a "rating case" forecast. This is distinct from a company's often-optimistic "management case." The rating case incorporates more conservative assumptions about growth and profitability to assess debt service capacity "through the cycle".¹² This process transforms forecasting from a mechanical exercise into a core part of the risk assessment.

```json
{
  "financial_forecasting_and_stress_testing": {
    "description": "Develops a forward-looking 'rating case' forecast and tests its resilience under a downside scenario.",
    "prompts": []
  }
}
```

### financial\_flexibility\_and\_liquidity

> This assesses the company's ability to meet near-term obligations and manage unexpected cash shortfalls. It involves analyzing the debt maturity profile, available liquidity sources, and covenant headroom under credit facilities.² A potential covenant breach is a significant credit event that can trigger defaults.

```json
{
  "financial_flexibility_and_liquidity": {
    "description": "Assesses the company's near-term liquidity position, debt maturity profile, and covenant headroom.",
    "prompts": []
  }
}
```

-----

## V. Synthesis, Rating, and Reporting

The final stage of the analysis involves integrating all prior findings, benchmarking the company against peers, and arriving at a defensible credit rating recommendation. The process is not a simple summation of factors but a structured judgment that often uses an "anchor and modifier" framework.¹¹ The combination of the **Business Risk** and **Financial Risk** profiles determines an "anchor" rating. This anchor is then adjusted up or down based on modifying factors like liquidity, financial policy, or structural features of a specific debt instrument. This two-step process mirrors the nuanced deliberations of a real rating committee.³

### peer\_analysis

> A company's credit metrics are only meaningful when placed in the context of its peers. This systematic comparison helps to normalize for industry-specific characteristics and highlights areas of relative strength or weakness.³

```json
{
  "peer_analysis": {
    "description": "Benchmarks the subject company against a group of relevant, publicly-rated peers.",
    "prompts": []
  }
}
```

### risk\_profile\_synthesis

> This is where the two main pillars of the analysis—Business Risk and Financial Risk—are formally combined to derive an initial, or "anchor," credit assessment.

| Business Risk Profile | Financial Risk Profile | | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | **Minimal** | **Modest** | **Intermediate** | **Significant** | **Aggressive** | **Highly Leveraged** |
| **Excellent** | aaa | aa | a | bbb | bb | b |
| **Strong** | aa | a | bbb | bb | b | b- |
| **Satisfactory** | a | bbb | bb | b+ | b- | ccc |
| **Fair** | bbb | bb | b+ | b | b- | ccc |
| **Weak** | bb | b+ | b | b- | ccc | cc |
| **Vulnerable** | b | b- | ccc | cc | c | c |
**Table 3: Illustrative Business & Financial Risk Scoring Matrix.** Modeled on the S\&P framework, this matrix provides a systematic approach for combining the qualitative business risk assessment with the quantitative financial risk assessment to determine an "anchor" credit profile. It visually demonstrates the core principle that a stronger business can support greater financial risk for a given rating level.¹¹

```json
{
  "risk_profile_synthesis": {
    "description": "Integrates the Business and Financial risk assessments to determine an 'anchor' credit profile.",
    "prompts": []
  }
}
```

### modifying\_factors\_and\_notching

> The anchor rating is adjusted for other material factors. A particularly strong or weak liquidity profile can warrant an adjustment. For specific debt instruments, recovery analysis determines whether the instrument rating should be at, above, or below the issuer's overall credit rating based on its security and seniority in the capital structure.¹⁹

```json
{
  "modifying_factors_and_notching": {
    "description": "Adjusts the anchor rating for other material factors like liquidity, financial policy, and instrument-specific features.",
    "prompts": []
  }
}
```

### rating\_recommendation

> This is the final, actionable output. It includes the recommended rating, a forward-looking outlook, and a concise rationale. The outlook (Stable, Positive, Negative) is a critical component, communicating the likely direction of the rating over the next 12-24 months and is based on the potential for identified risks or opportunities to materialize.⁸

```json
{
  "rating_recommendation": {
    "description": "States the final rating recommendation, outlook, and a concise summary of the rating rationale.",
    "prompts": []
  }
}
```

### credit\_report\_generation

> This final object provides prompts to assemble the full narrative report from the preceding analytical components, ensuring a professional and comprehensive final deliverable consistent with industry standards.⁴

```json
{
  "credit_report_generation": {
    "description": "Assembles the full narrative credit report from the completed analytical sections.",
    "prompts": []
  }
}
```

```
```
