# Universal Ingestor Pipeline

The `core/data_processing/` directory houses the "Universal Ingestor," Adam's centralized data ETL (Extract, Transform, Load) pipeline.

## 🎯 Mission
**"Garbage in, Garbage out."** The Ingestor ensures that only clean, normalized, and verified data reaches the reasoning engine.

## 🔑 Key Components

### 1. Universal Ingestor (`universal_ingestor.py`)
The main entry point. It accepts raw files (PDFs, HTML, JSON) or text and orchestrates the cleaning process.
*   **Features:** PII redaction (via Presidio), format normalization, and metadata tagging.

### 2. Semantic Conviction (`semantic_conviction.py`)
A module that scores data reliability. It uses embeddings to compare new information against known "Golden Truths" (e.g., verified SEC filings).
*   **Input:** A data snippet.
*   **Output:** A 0-100% confidence score.

### 3. Conviction Scorer (`conviction_scorer.py`)
A heuristic-based scorer that evaluates the source credibility (e.g., Bloomberg > Reddit).

## 🚀 Usage

```python
from core.data_processing.universal_ingestor import UniversalIngestor

ingestor = UniversalIngestor()
clean_data = ingestor.process_file("path/to/annual_report.pdf")
print(clean_data.metadata.conviction_score)
```

## ⚠️ Extension Rules
*   **Idempotency:** Processing the same file twice should yield the same result.
*   **Non-Destructive:** Always preserve the raw source link/hash in the metadata.
