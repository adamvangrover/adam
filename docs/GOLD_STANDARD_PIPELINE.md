# Gold Standard Data Pipeline

The Gold Standard Data Pipeline is a key component of the Adam v23.5 "Adaptive System". Its purpose is to ingest, standardize, and certify data from across the repository, converting it into high-quality, machine-readable knowledge artifacts.

## Overview

The pipeline operates on the principle of "Universal Ingestion". It scans the entire repository for valuable information—reports, prompts, code documentation, newsletters, and raw data—and processes it through a rigorous scrubbing and conviction assessment stage.

### Key Components

1.  **Universal Ingestor** (`core/data_processing/universal_ingestor.py`):
    *   Recursively scans directories.
    *   Identifies artifact types (Reports, Prompts, Data, etc.).
    *   Standardizes content into a common schema.

2.  **Gold Standard Scrubber** (`core/data_processing/gold_standard_scrubber.py`):
    *   **Cleaning**: Removes artifacts, fixes encoding, standardizes whitespace.
    *   **Metadata Extraction**: Automatically extracts keys, entities, and structure metrics.
    *   **Conviction Assessment**: Assigns a `conviction_score` (0.0 - 1.0) based on data quality, completeness, and structure.

3.  **Knowledge Artifacts** (`data/gold_standard/knowledge_artifacts.jsonl`):
    *   The final output is a JSONL file where every line is a standardized `GoldStandardArtifact`.
    *   This file serves as the "Long Term Memory" for the system.

## Usage

To run the pipeline and generate the UI data:

```bash
python scripts/generate_ui_data.py
```

This script will:
1.  Initialize the `UniversalIngestor`.
2.  Scan `core/libraries_and_archives`, `prompt_library`, `data`, and `docs`.
3.  Process and score all findings.
4.  Output `data/gold_standard/knowledge_artifacts.jsonl`.
5.  Generate `showcase/data/ui_data.json` and `showcase/js/mock_data.js` for the frontend.

## Schema

Each artifact in the Gold Standard dataset follows this structure:

```json
{
  "id": "uuid",
  "source_path": "path/to/original/file",
  "type": "report|prompt|data|...",
  "title": "Artifact Title",
  "content": "...",
  "metadata": { ... },
  "conviction_score": 0.95,
  "ingestion_timestamp": "ISO-8601"
}
```

## Future Enhancements

*   **AI Review**: Integrate an LLM call to semantically verify the content of high-value artifacts.
*   **Vector Embedding**: Automatically generate embeddings for each artifact during ingestion.
