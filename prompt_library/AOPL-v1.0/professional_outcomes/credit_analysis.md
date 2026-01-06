# Guide to Corporate Credit Risk Analysis using the Prompt Library

## Introduction

Welcome, financial analyst! This guide is designed to help you leverage our comprehensive JSON prompt library to conduct a thorough and standardized corporate credit risk review. The goal of this library is to provide a structured framework for your analysis, ensuring all critical aspects of credit risk are considered consistently and efficiently. By using these structured prompts, you can enhance the quality, depth, and consistency of your credit assessments, whether for a new underwriting, an annual review, or ongoing monitoring.

---

## Overview of the Prompt Library JSON Structure

The provided JSON file is the backbone of your analysis. It's organized into several key sections:

* **`prompt_metadata`**: Contains general information about the prompt library version and author.
* **`report_specifications`**: Outlines the intended audience, tone, and format for the output.
* **`core_analysis_areas`**: This is the heart of the library. It's an array of individual prompt objects, each designed to tackle a specific part of the credit analysis. Each prompt has an `id`, `title`, `instructions`, and a crucial list of `key_considerations`.
* **`data_requirements_general`**: Lists the typical data and documents you'll need for a comprehensive review.
* **`expert_guidance_notes_general`**: Provides high-level best practices for using the prompts effectively.

Your main focus will be on the `core_analysis_areas`, as these provide the building blocks for your credit memorandum.

---

## How to Use This Guide

This document will walk you through the typical workflow of a corporate credit review. Each step in the process corresponds to a specific section of a standard credit write-up. For each step, this guide will:

1.  **Identify the relevant prompt(s)** from the library by its `prompt_title` and `(prompt_id)`.
2.  **Summarize the objective** of that analytical section.
3.  **List key questions** you should answer, based on the `key_considerations` in the prompt, to build your analysis.

Think of this guide as a roadmap and the prompt library as your toolkit.

---

## Step-by-Step Credit Review Walkthrough

Here is a breakdown of a standard credit analysis, mapping each stage to the relevant prompts in the library.

### I. Company and Business Profile Analysis

* **Objective**: To establish a foundational understanding of the company's business model, operational scale, and market presence.
* **Relevant Prompt(s) from Library**: Company Overview and Business Profile (`company_overview_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * What are the company's core business activities, and how does it actually make money?
    * What are its main products or services?
    * What is the scale of its operations (consider revenue, total assets, number of employees)?
    * What is its geographic footprint? Is it diversified or concentrated?
    * Who are its most critical customers and suppliers? Is there any concentration risk?
    * What is its ownership structure (e.g., public, private, a subsidiary)?

### II. Industry and Competitive Landscape Assessment

* **Objective**: To evaluate the external environment in which the company operates, including industry trends, risks, and the intensity of competition.
* **Relevant Prompt(s) from Library**: Industry Analysis and Competitive Landscape (`industry_analysis_competitive_landscape_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * How large is the market, and what are its growth prospects and key trends (e.g., technology, consolidation)?
    * What are the primary drivers of success in this industry?
    * How intense is the competition (consider a Porter's Five Forces analysis)? Who are the major players?
    * What is the company's market position (e.g., leader, niche player), and what are its sustainable competitive advantages?
    * Are there significant barriers to entry that protect the company?
    * What are the key industry-wide risks (e.g., regulatory, cyclicality, technological disruption)?

### III. Financial Statement Deep Dive

* **Objective**: To dissect the company's financial health and performance through a detailed analysis of its financial statements.
* **Relevant Prompt(s) from Library**: Financial Statement Analysis (`financial_statement_analysis_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * **Profitability**: How profitable is the company? Analyze trends in Gross, EBITDA, and Net Margins. How do its returns (ROA, ROE, ROIC) look over time and against peers?
    * **Leverage**: How is the company capitalized? Assess its debt burden using ratios like Debt-to-EBITDA and Debt-to-Capital. Is the capital structure appropriate?
    * **Liquidity**: Can the company meet its short-term obligations? Analyze the Current and Quick Ratios. How efficiently does it manage working capital (DSO, DIO, DPO)?
    * **Coverage**: How easily can the company service its debt? Focus on Interest Coverage (EBITDA/Interest) and Debt Service Coverage Ratios.
    * **Efficiency**: How effectively are assets being used to generate sales? Look at Asset Turnover ratios.
    * **Cash Flow**: Is the company generating cash? Analyze the quality and trends of cash flow from operations and determine its Free Cash Flow (FCF) generation capacity. How does FCF relate to its total debt?

### IV. Performance Evaluation

* **Objective**: To assess the company's historical performance and the credibility of its future financial projections.
* **Relevant Prompt(s) from Library**: Historical and Projected Performance Evaluation (`performance_evaluation_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * What have been the historical drivers of revenue and profitability growth?
    * How volatile have earnings and cash flows been in the past?
    * If management has provided projections, what are the key assumptions? Are they realistic when compared to historical performance and the industry outlook?
    * What are the primary risks to the company achieving its financial targets?

### V. Management and Governance Assessment

* **Objective**: To evaluate the capability and credibility of the management team and the strength of the company's corporate governance framework.
* **Relevant Prompt(s) from Library**: Management and Governance Assessment (`management_assessment_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * How experienced and deep is the management team? What is their track record?
    * Is the corporate strategy clear, credible, and well-executed?
    * What is the company's financial policy regarding risk, leverage, and shareholder returns?
    * Are there any corporate governance red flags (e.g., lack of board independence, related-party transactions, poor disclosure)?

### VI. Strengths and Weaknesses Summary

* **Objective**: To distill the entire analysis into a balanced, concise summary of the key factors supporting and detracting from the company's creditworthiness.
* **Relevant Prompt(s) from Library**: Credit Strengths and Weaknesses Summary (`strengths_weaknesses_summary_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * What are the top 3-5 factors that support the company's ability to repay its debt (e.g., strong market position, low leverage, high margins)?
    * What are the top 3-5 factors that represent a risk to repayment (e.g., high customer concentration, volatile cash flows, competitive threats)?

### VII. Risk Assessment and Probability of Default

* **Objective**: To formally assess the likelihood of the company defaulting on its obligations by synthesizing quantitative and qualitative factors.
* **Relevant Prompt(s) from Library**: Probability of Default (PD) Assessment (`probability_of_default_rating_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * Based on the financial and business analysis, what is the overall risk profile?
    * What key quantitative metrics (e.g., leverage, coverage) and qualitative factors (e.g., competitive strength, industry risk) are driving the default risk?
    * How would the company's ability to pay be affected by a downturn or stress scenario?
    * What is the final conclusion on the probability of default (e.g., Low, Medium, High) and what is the core rationale?

### VIII. Covenant Analysis

* **Objective**: To understand the contractual protections in the debt agreements and assess the company's ability to remain in compliance.
* **Relevant Prompt(s) from Library**: Covenant Analysis (`covenant_analysis_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * What are the key financial covenants (e.g., Maximum Debt/EBITDA, Minimum Interest Coverage)?
    * What is the current level of compliance and how much headroom or "cushion" does the company have?
    * How sensitive is the covenant headroom to a decline in EBITDA?
    * What are the consequences of a covenant breach?

### IX. Structural Considerations

* **Objective**: To analyze risks and support mechanisms arising from the company's position within a larger corporate group.
* **Relevant Prompt(s) from Library**: Parent/Subsidiary Linkage and Group Support Assessment (`parent_subsidiary_linkage_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * Is the company a strategically important part of a larger, stronger (or weaker) parent organization?
    * Are there any explicit forms of support, such as parental guarantees or cross-default provisions?
    * Is there a history of the parent supporting its subsidiaries?
    * Conversely, could problems at the parent or a sister company negatively impact the entity being analyzed (contagion risk)?

### X. External Factors (Macroeconomic, Country, ESG)

* **Objective**: To assess risks originating from outside the company and its industry, including macroeconomic, political, and ESG factors.
* **Relevant Prompt(s) from Library**:
    * Country and Macroeconomic Risk Assessment (`country_macroeconomic_risk_prompt`)
    * ESG (Environmental, Social, Governance) Credit Factors Analysis (`esg_credit_factors_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * In what countries does the company operate, and what are the associated political, economic, and currency risks?
    * How would changes in GDP growth, inflation, or interest rates impact the company?
    * What are the most *material* Environmental, Social, and Governance risks for this specific company? (e.g., carbon transition risk for an oil company, labor relations for a retailer).
    * How are these ESG risks being managed, and could they have a tangible impact on financial performance?

### XI. Credit Outlook and Rating Triggers

* **Objective**: To provide a forward-looking view on the likely direction of credit quality and define specific events that would cause a re-evaluation.
* **Relevant Prompt(s) from Library**:
    * Credit Outlook Assessment (`credit_outlook_assessment_prompt`)
    * Rating Triggers (Upgrade/Downgrade Scenarios) (`rating_triggers_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * Over the next 12-24 months, is the company's credit profile likely to improve, deteriorate, or remain stable? Why?
    * What specific, measurable events would trigger a rating upgrade (e.g., Debt/EBITDA sustained below 2.0x)?
    * What specific events would trigger a downgrade (e.g., loss of a major customer, a large debt-funded acquisition)?

### XII. Regulatory Considerations

* **Objective**: To analyze the credit from a regulatory perspective, particularly for bank analysts dealing with shared credits.
* **Relevant Prompt(s) from Library**: Shared National Credit (SNC) Regulatory Rating Analysis (`snc_regulatory_rating_prompt`)
* **Key Questions and Areas of Focus for the Analyst**:
    * What is the primary source of repayment, and is it reliable?
    * Does the company generate enough cash flow from operations to service all its debt obligations in a timely manner?
    * Are there any well-defined weaknesses that jeopardize repayment?
    * How does the company's profile map to regulatory definitions like "Pass," "Special Mention," or "Substandard"?

---

## Utilizing Full Report Structure Prompts

Beyond the individual analysis blocks, the library includes prompts to help you assemble complete reports and gather information:

* **`underwriting_memo_structure_prompt`**: Use this as a master template when you are analyzing a new loan or transaction. It provides a comprehensive outline for a credit memo, referencing the individual analytical prompts you've just learned about for each section.
* **`annual_review_monitoring_update_prompt`**: This prompt provides a tailored structure for periodic reviews. It focuses on performance since the last update, covenant compliance, and any changes to the company's risk profile.
* **`due_diligence_checklist_credit_prompt`**: This is an excellent tool to use at the *beginning* of your process. It generates a comprehensive checklist to ensure you request all the necessary business, financial, and legal information from the company.

---

## General Guidance

As you use the prompt library, keep these expert tips in mind:

* **Be Specific**: Always clearly define the company and the time periods you are analyzing.
* **Context is Key**: Tailor your analysis to the specific reason for the review (e.g., new loan, annual review, event-driven update).
* **Justify Everything**: Clearly link your data and analysis to your conclusions. The "why" is just as important as the "what."
* **Distinguish Fact from Opinion**: Be clear when you are stating historical facts versus providing forward-looking projections or opinions.
* **Define Your Metrics**: Ensure that all financial ratios are clearly defined and calculated consistently.

---

## Conclusion

This guide and the accompanying JSON prompt library provide a powerful combination for producing high-quality, comprehensive, and consistent corporate credit risk analysis. By following the structured steps and asking the key questions outlined here, you can be confident that your reviews are thorough and well-supported. Happy analyzing!
