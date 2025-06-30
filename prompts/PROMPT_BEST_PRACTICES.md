# Best Practices for Prompting and Prompt Library (`/prompts`)

## 1. Introduction

This document outlines best practices for crafting effective prompts, particularly for generating financial analysis, reports, and insights using Large Language Models (LLMs) or advanced AI agent systems like the Adam platform. The goal of a well-designed prompt is to achieve consistent, accurate, and high-quality outputs, minimizing ambiguity and maximizing the utility of the AI's capabilities.

The `/prompts` directory serves as a library of structured prompt templates. These templates are designed to be both human-readable (for understanding and modification) and machine-parsable (for potential automation and integration into AI workflows).

## 2. Core Principles of Effective Prompting

Effective prompting is an art and a science. Here are fundamental principles:

*   **Clarity and Specificity:**
    *   Be explicit and unambiguous in your instructions. Avoid vague language.
    *   Clearly define the scope of the task. What exactly do you want the AI to do? What should it *not* do?
    *   Provide precise details and constraints.
*   **Context is Key:**
    *   Supply sufficient background information relevant to the query.
    *   Reference specific data, documents (e.g., from `core/libraries_and_archives/` or `data/` in this repo), or previous conversation points if applicable.
    *   The more relevant context the AI has, the better its output will align with your expectations.
*   **Structured Format:**
    *   Organize complex prompts into logical sections. Our JSON prompt templates exemplify this.
    *   A structured approach helps both humans in crafting prompts and AI in interpreting them. It also facilitates easier updates and maintenance of prompts.
*   **Define the Persona/Role:**
    *   Instruct the AI on the role or persona it should adopt (e.g., "You are a senior financial analyst," "You are a risk manager," "You are a concise market commentator"). This influences tone, style, and depth of analysis.
*   **Specify Output Format:**
    *   Clearly define the desired structure (e.g., Markdown, JSON, specific sections, bullet points, tables), length, and writing style (e.g., formal, informal, objective, persuasive).
*   **Iterative Refinement:**
    *   Prompting is often an iterative process. Your first prompt may not yield the perfect result.
    *   Be prepared to experiment, analyze the AI's output, and refine your prompt based on the results. Small changes can lead to significant improvements.
*   **Break Down Complex Tasks:**
    *   If a task is highly complex, consider breaking it into smaller, sequential prompts. This can lead to better quality outputs for each sub-task.

## 3. Key Components of a High-Impact Prompt (using our JSON structure)

The JSON templates in the `/prompts` library provide a robust framework. Key components include:

*   **`prompt_metadata`**:
    *   **Purpose:** Tracks essential information about the prompt itself.
    *   **Fields:** `prompt_id`, `prompt_version`, `creation_date`, `description`, `author`.
    *   **Benefit:** Useful for version control, understanding prompt evolution, and collaborative prompt engineering.
*   **`report_specifications`**:
    *   **Purpose:** Defines the high-level parameters and desired characteristics of the output.
    *   **Fields:** `report_title`, `time_horizon`, `target_audience`, `output_format`, `tone_and_style`, and task-specific parameters (e.g., `company_name`, `sector_name`).
    *   **Benefit:** Sets clear expectations for the AI regarding the final deliverable.
*   **`core_analysis_areas`**:
    *   **Purpose:** Breaks down the main request into logical, structured sections. This is the heart of the prompt.
    *   **Structure:** Typically an array of objects, each representing a section with `section_id`, `section_title`, `instructions` (for the AI), and `key_considerations` (specific points, questions, or data to address). Sub-sections can be nested for further granularity.
    *   **Benefit:** Ensures comprehensive coverage of the topic and guides the AI's analytical flow.
*   **`data_requirements`**:
    *   **Purpose:** Lists the types of input data, documents, or access needed for the AI to fulfill the prompt effectively.
    *   **Benefit:** Helps in preparing for the prompt execution and highlights dependencies. For an integrated system like Adam, this might map to specific data retrieval agents or knowledge base queries.
*   **`expert_guidance_notes`**:
    *   **Purpose:** Provides additional tips, best practices, or constraints for the AI to enhance output quality.
    *   **Benefit:** Captures nuanced instructions that don't fit elsewhere, akin to giving expert advice to an analyst.

## 4. Best Practices for Financial Prompts (Adam System Context)

When prompting in a financial domain, especially within a sophisticated AI system:

*   **Leveraging Specialized Agents:**
    *   Design prompts that can be conceptually (or actually, in an advanced system) decomposed and routed to specialized agents (e.g., a `MacroeconomicAnalysisAgent`, `FundamentalAnalystAgent`, `RiskAssessmentAgent`).
    *   Structure sections in your prompt to align with the kind of analysis a specialized agent would perform.
*   **Quantitative Data Focus:**
    *   Prompt for specific quantitative data, ratios, trends, and calculations.
    *   Example: "Calculate the 3-year CAGR for revenue," "Compare the P/E ratios of Company A and Company B."
*   **Risk, Nuance, and Balanced Views:**
    *   Explicitly ask for identification of risks, assumptions, uncertainties, and limitations.
    *   Encourage a balanced perspective, including both pros and cons, or bull and bear cases.
*   **Referencing Internal Data/Knowledge:**
    *   If the AI system has access to internal knowledge bases (like this repository's `core/libraries_and_archives/` or `data/` folders), craft prompts to leverage this.
    *   Example: "Using the Q1 2025 Outlook report (<code>core/libraries_and_archives/reports/Q1 2025 and Full Year Outlook: Navigating a Bifurcated Market.json</code>), summarize the key geopolitical risks identified."
    *   Be specific about filenames or data identifiers if possible.
*   **Chain-of-Thought/Step-by-Step Reasoning:**
    *   For complex analyses, encourage the AI to "think step by step" or outline its reasoning process. This can improve the quality and transparency of the output.
    *   Example: "First, identify the key financial ratios for liquidity. Second, calculate these for the past 3 years. Third, analyze the trend and compare to industry averages."
*   **Handling Ambiguity in Financial Language:**
    *   Finance has terms that can be ambiguous. Be precise (e.g., specify "Net Income" vs. "Adjusted Net Income").
*   **Time Sensitivity:**
    *   Clearly specify dates, reporting periods (e.g., "latest fiscal quarter," "TTM"), and time horizons for forecasts.

## 5. Using the Prompt Library (`prompts/` directory)

*   **Understanding the Templates:** Familiarize yourself with the structure of the JSON prompt templates. Each file is a self-contained request for a specific type of report or analysis.
*   **Filling Placeholders:** Templates use placeholders like `[Specify Company Name]` or `[Current Date]`. Replace these with the actual values relevant to your specific request before using the prompt.
*   **Adaptation:**
    *   Modify existing templates to suit slightly different needs. You can add, remove, or alter sections and `key_considerations`.
    *   Use the existing templates as a foundation for creating entirely new prompts for different tasks, maintaining a consistent structure.
*   **Contribution:** If new, generally useful prompt types are developed, consider adding them to the library using the established JSON format.

## 6. Troubleshooting / Improving Prompts

If the AI's output isn't what you expected:

*   **Increase Specificity:** Is any part of your prompt vague or open to multiple interpretations? Add more detail.
*   **Add More Context:** Did the AI lack crucial background information?
*   **Simplify the Request:** Is the prompt too complex or asking for too many things at once? Try breaking it down.
*   **Check for Conflicting Instructions:** Ensure different parts of your prompt don't give contradictory guidance.
*   **Refine `key_considerations`:** Are they precise enough? Do they guide the AI effectively towards the desired details?
*   **Adjust Persona/Tone:** If the style is off, reiterate or refine the persona and tone instructions.
*   **Examine Examples:** If you provided examples of desired output, ensure they are clear and consistent with your instructions.

## 7. Conclusion

A systematic and thoughtful approach to prompting is crucial for unlocking the full potential of advanced AI systems in financial analysis. By using clear, specific, context-rich, and well-structured prompts, we can guide AI to produce more accurate, relevant, and valuable insights. The `prompts/` library provides a starting point and a framework for developing and managing high-impact prompts within the Adam ecosystem. Continuous learning and iterative refinement of prompting skills will be key to maximizing the benefits of this technology.
