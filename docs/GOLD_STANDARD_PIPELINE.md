# Gold Standard Data Pipeline

## Overview
The "Gold Standard" Data Pipeline is a robust ingestion and standardization system designed to convert heterogeneous data sources (reports, newsletters, prompts, raw data) into a high-quality, unified format. This ensures that all agents, knowledge graphs, and UI components have access to a single, verified source of truth.

## Architecture

### 1. Universal Ingestor
Located at `core/data_processing/universal_ingestor.py`, this module acts as the entry point.
- **Scans:** Recursively scans specified directories (`core/libraries_and_archives`, `prompt_library`, `data`).
- **Parses:** Handles multiple file formats (`.json`, `.jsonl`, `.md`, `.txt`).
- **Standardizes:** Converts all inputs into `GoldStandardArtifact` objects.

### 2. Gold Standard Artifact Schema
Every ingested item is normalized to the following schema:
```json
{
  "id": "uuid-v4",
  "source_path": "path/to/original/file",
  "type": "report|newsletter|prompt|data|code_doc",
  "title": "Extracted Title",
  "content": "Raw content or parsed JSON",
  "metadata": { ... },
  "conviction_score": 0.95,
  "ingestion_timestamp": "ISO-8601"
}
```

### 3. Storage
The processed data is saved to `data/gold_standard/knowledge_artifacts.jsonl`. This file serves as the "Long Term Memory" for the system.

### 4. UI Integration
The script `scripts/generate_ui_data.py` utilizes the Universal Ingestor to populate the static UI data (`showcase/js/mock_data.js`). This ensures that the "Showcase" experience is always grounded in the actual repository content, with graceful fallbacks to synthetic data only when necessary.

## Usage

### Running the Pipeline
To regenerate the gold standard dataset and update the UI:
```bash
python scripts/generate_ui_data.py
```

### Extending the Pipeline
To add new data sources or formats, modify `core/data_processing/universal_ingestor.py`:
- Add a new `_process_extension` method.
- Update the `scan_directory` logic to include new paths.

## Conviction Scoring
Currently, all repository data is assigned a default high conviction score (0.95) as it represents the "Brain" of the system. Future iterations can implement an LLM-based `assess_conviction` method to dynamically score incoming external data.
