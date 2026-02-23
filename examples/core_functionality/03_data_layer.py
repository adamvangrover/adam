"""
Example 03: Data Layer Standalone
---------------------------------
Demonstrates the Data Layer (Ingestion & ETL) operating independently.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.system.provenance_logger import ProvenanceLogger, ActivityType

# Mock Ingestor to demonstrate pattern without needing heavy PDF libs
class DataLayerIngestor:
    def __init__(self):
        self.logger = ProvenanceLogger()

    def ingest_document(self, filepath: str, source_type: str):
        print(f"[{source_type}] Ingesting: {filepath}")

        # 1. Read (Mock)
        raw_content = "Simulated content from file..."

        # 2. Chunk
        chunks = [raw_content[i:i+10] for i in range(0, len(raw_content), 10)]
        print(f"Generated {len(chunks)} chunks.")

        # 3. Log Provenance
        self.logger.log_activity(
            agent_id="UniversalIngestor",
            activity_type=ActivityType.INGESTION,
            input_data={"filepath": filepath, "size": len(raw_content)},
            output_data={"chunks_count": len(chunks)},
            data_source=source_type,
            capture_full_io=True
        )

        return chunks

if __name__ == "__main__":
    print(">>> Starting Data Layer ETL...")
    ingestor = DataLayerIngestor()

    ingestor.ingest_document("data/10k/aapl_2025.pdf", "SEC_EDGAR")
    ingestor.ingest_document("data/news/bloomberg_wire.txt", "BloombergTerminal")

    print(">>> Ingestion complete.")
