# Adam v24.0 Integration Guide: The Alphabet Ecosystem

## Overview

Adam v24.0 represents a strategic shift towards a cloud-native, AI-first architecture leveraging the **Alphabet Ecosystem**. This integration unifies **Google Vertex AI**, **Gemini 1.5 Pro**, and **BigQuery** to create a scalable, multimodal financial intelligence platform.

## Architecture Components

### 1. The Brain: Gemini 1.5 Pro (via Vertex AI)
Adam now utilizes Gemini 1.5 Pro as its primary cognitive engine.
*   **Role**: Deep Qualitative Analysis, Multimodal Reasoning (Charts/Graphs), and Long-Context Synthesis (1M+ tokens).
*   **Integration**: `core/analysis/gemini_analyzer.py` wraps the Vertex AI SDK.
*   **Key Feature**: "Chain of Thought" prompting is enforced to ensure reasoning precedes structured JSON output.

### 2. The Memory: Vertex AI Vector Search
*   **Role**: Episodic Memory and RAG (Retrieval Augmented Generation).
*   **Integration**: `core/memory/episodic_memory.py` connects to `VertexVectorStore`.
*   **Function**: Allows agents to recall past analyses and retrieve relevant historical documents based on semantic similarity.

### 3. The Nervous System: Pub/Sub & Async Execution
*   **Role**: Decoupling long-running analysis tasks from the main agent loop.
*   **Integration**: `core/infrastructure/bigquery_connector.py` (PubSubConnector stub).
*   **Flow**:
    1.  Agent requests analysis.
    2.  Request published to `analysis-topic`.
    3.  Worker process (Gemini Analyzer) consumes message.
    4.  Result stored in BigQuery and notified via callback.

### 4. The Warehouse: BigQuery
*   **Role**: Structured storage for all financial insights, risk metrics, and agent logs.
*   **Integration**: `core/infrastructure/bigquery_connector.py`.

### 5. The Future: Quantum-Native Risk
*   **Role**: Monte Carlo simulations enhanced by Quantum Computing concepts (Amplitude Estimation).
*   **Integration**: `core/risk_engine/quantum_monte_carlo.py`.
*   **Status**: Currently running on a classical simulator (`SimulatorBackend`) with interfaces ready for QPU drop-in.

## Configuration

To enable the Alphabet Ecosystem:

1.  **Environment Variables**:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
    export PROJECT_ID="your-gcp-project-id"
    export VERTEX_LOCATION="us-central1"
    export GEMINI_API_KEY="your-api-key" (Optional if using Vertex AI directly)
    ```

2.  **Dependencies**:
    Ensure `google-cloud-aiplatform`, `google-cloud-bigquery`, and `google-generativeai` are installed.

## Usage Example

```python
from core.analysis.gemini_analyzer import get_gemini_analyzer

analyzer = get_gemini_analyzer()
result = await analyzer.analyze_report(
    report_text="...",
    context={"ticker": "GOOGL", "period": "2023-Q4"}
)
print(result.strategic_insights)
```
