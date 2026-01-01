# Alphabet Ecosystem Integration Guide

## Overview
This guide details the integration of Google Cloud's **Gemini 1.5 Pro**, **Vertex AI**, and **Pub/Sub** into the Adam architecture. This expansion transforms Adam from a text-based analyst into a multimodal, asynchronous cognitive engine.

## Architecture Components

### 1. Gemini Financial Report Analyzer
Located in `core/analysis/gemini_analyzer.py`, this component utilizes Gemini 1.5 Pro's massive context window (up to 2M tokens) to digest entire annual reports (10-K) in a single pass.

**Key Features:**
*   **Chain-of-Thought Prompting:** Forces the model to generate a "Thinking Process" trace before outputting the final JSON, reducing hallucination rates in financial data extraction.
*   **Structured Output:** Strictly typed Pydantic models (`RiskFactor`, `StrategicInsight`) ensure downstream agents can consume the data without parsing errors.
*   **Multimodal Vision:** Can accept paths to chart images extracted from PDFs and generate analysis.

### 2. Vertex AI Infrastructure
The system interacts with Google Cloud via the `LLMPlugin` (`core/llm_plugin.py`), which now includes a `VertexLLM` adapter.

*   **Embeddings:** `get_embedding()` maps text to high-dimensional vectors for RAG (Retrieval Augmented Generation).
*   **Asynchronous Execution:** Heavy analysis tasks are offloaded to `asyncio` executors to prevent blocking the main agent orchestrator loop.

### 3. Data Warehousing (BigQuery)
A standardized interface `BigQueryConnector` (`core/infrastructure/bigquery_connector.py`) allows agents to dump structured insights directly into an enterprise data warehouse.

## Configuration

To enable these features, ensure your `.env` file or environment variables are set:

```bash
GEMINI_API_KEY="your-google-ai-studio-key"
# Optional for Vertex AI Enterprise
GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
VERTEX_PROJECT_ID="your-project-id"
VERTEX_LOCATION="us-central1"
```

## Usage Example

```python
from core.analysis.gemini_analyzer import get_gemini_analyzer

async def run_analysis():
    analyzer = get_gemini_analyzer()
    report_text = "..." # Load your 10-K text here

    # Run Deep Qualitative Analysis
    result = await analyzer.analyze_report(report_text, context={"ticker": "GOOGL", "period": "2023-10K"})

    print(f"Strategic Confidence: {result.strategic_insights[0].confidence_score}")
    print(f"Top Risk: {result.risk_factors[0].description}")
```

## Troubleshooting
*   **ImportError: google.generativeai**: Run `pip install google-generativeai`.
*   **403 Permission Denied**: Check your Vertex AI API enablement in Google Cloud Console.
