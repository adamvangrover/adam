# Guide to Regulatory Rating Analysis using the Prompt Library

## Introduction

This guide is for credit professionals who need to assign a regulatory rating to a corporate credit facility, such as for the Shared National Credit (SNC) program. The prompts in the accompanying JSON file (`regulatory_rating.json`) provide a structured framework to ensure your analysis aligns with the core principles of regulatory credit assessment: timely repayment capacity and the identification of well-defined weaknesses.

---

## How to Use the Prompts

The `regulatory_rating.json` file contains a single `core_analysis_area` focused on this task. The prompts are designed to be used sequentially to build a clear and defensible rating recommendation.

### Regulatory Rating Workflow

#### 1. Analyze the Primary Source of Repayment

*   **Objective**: To determine the primary source of cash to repay the loan and assess its reliability.
*   **Relevant Prompt**: `repayment_source_analysis` (Task ID: `RR01`)
*   **Analyst Focus**:
    *   Is repayment expected from operating cash flow, sale of assets, or refinancing?
    *   How dependable is this source over the life of the loan?
    *   A "Pass" credit typically has a reliable and ongoing source of repayment from operations.

#### 2. Assess Cash Flow Adequacy

*   **Objective**: To verify that the company's cash flow is sufficient to meet all its debt obligations.
*   **Relevant Prompt**: `cash_flow_adequacy` (Task ID: `RR02`)
*   **Analyst Focus**:
    *   Using conservative assumptions, does the company generate enough cash to cover both interest and principal payments as they come due?
    *   This is a forward-looking view. Historical performance is a guide, but future capacity is key.
    *   Inability to service debt from normal operations is a significant weakness.

#### 3. Identify Well-Defined Weaknesses

*   **Objective**: To identify any specific, material issues that jeopardize the timely repayment of the loan.
*   **Relevant Prompt**: `weakness_identification` (Task ID: `RR03`)
*   **Analyst Focus**:
    *   This is the core of what separates a "Pass" from a criticized rating.
    *   Look for issues like:
        *   A sustained negative trend in financial performance.
        *   Breaches of financial covenants.
        *   Over-reliance on asset sales or refinancing to meet obligations.
        *   Poor management or flawed business strategy.
    *   A "Special Mention" rating is assigned when such weaknesses are present, but they have not yet reached a level where default is imminent.

#### 4. Synthesize and Recommend a Rating

*   **Objective**: To combine the findings into a final rating and justification.
*   **Relevant Prompt**: `rating_recommendation_synthesis` (Task ID: `RR04`)
*   **Analyst Focus**:
    *   **Pass**: The company has a sound primary source of repayment and sufficient cash flow to service its debt. There are no well-defined weaknesses that jeopardize repayment.
    *   **Special Mention**: The company has potential weaknesses that, if not corrected, could result in a deterioration of repayment prospects. The asset is currently protected, but the risk is elevated.
    *   **Substandard**: The company has well-defined weaknesses that jeopardize the orderly repayment of the debt. There is a distinct possibility that the bank will sustain some loss if the deficiencies are not corrected.

---

## Conclusion

By following this structured approach, you can ensure that your regulatory rating recommendations are consistent, well-documented, and aligned with regulatory expectations. The prompts are designed to help you focus on the most critical factors and build a clear, evidence-based rationale for your conclusion.
