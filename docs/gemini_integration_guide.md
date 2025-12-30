# Google Gemini Integration Guide

## Overview
This document details the integration of Google Gemini (via `google-generativeai`) into the Adam financial system. The integration aims to provide "Deep Dive" qualitative analysis of financial reports, complementing the existing quantitative models (DCF, Ratios).

## Architecture

### 1. `LLMPlugin` Enhancement
The `core.llm_plugin.LLMPlugin` has been upgraded to support the `gemini` provider.
- **Model**: Default is `gemini-1.5-pro` (configurable).
- **Features**:
    - `generate_text`: Standard text generation.
    - `generate_structured`: Native structured output using Pydantic schemas (Prompt-as-Code).
    - `get_embedding`: Support for `models/embedding-001`.
    - `get_context_length`: 1M tokens for 1.5 Pro.

### 2. `GeminiFinancialReportAnalyzer`
Located in `core.analysis.gemini_analyzer.py`, this component encapsulates the logic for deep financial analysis.
- **Inheritance**: Implements `BaseFinancialAnalyzer` for future alignment.
- **Capabilities**:
    - **Risk Analysis**: Severity, Time Horizon, Mitigation.
    - **Strategic Insights**: Confidence scoring, Sentiment.
    - **Competitor Analysis**: Identification of competitive dynamics.
    - **ESG Metrics**: Qualitative scoring of Environmental, Social, and Governance factors.
- **Prompting**: Utilizes "Chain of Thought" prompting to encourage reasoning before output generation.

### 3. `FundamentalAnalystAgent` Integration
The `FundamentalAnalystAgent` now optionally initializes the `GeminiFinancialReportAnalyzer`.
- **Workflow**:
    1.  Fetch quantitative data (A2A).
    2.  Calculate ratios and valuations.
    3.  **New**: If qualitative data exists, pass it to Gemini for async analysis.
    4.  Merge Gemini insights (Risks, ESG, Sentiment) into the final `analysis_summary`.

## Configuration

To enable Gemini, ensure the following in your environment (`.env`) or `settings.yaml`:

```bash
GOOGLE_API_KEY=your_api_key_here
```

In `config/llm_plugin.yaml` (optional override):
```yaml
provider: gemini
gemini_model_name: gemini-1.5-pro
```

## Future Roadmap

1.  **Multimodal Analysis**: The `BaseFinancialAnalyzer` includes an interface for `analyze_image`. Future updates will implement chart and table parsing using Gemini Vision.
2.  **Vector Search**: The `get_embedding` method paves the way for storing financial report embeddings in a vector database for semantic search.
3.  **Agent Expansion**: Other agents (e.g., `RiskAssessmentAgent`) can reuse the `GeminiFinancialReportAnalyzer` or inherit from `BaseFinancialAnalyzer` for specialized tasks.
