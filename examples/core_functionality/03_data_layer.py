"""
Example 03: Data Layer Standalone
---------------------------------
Demonstrates the Data Layer (Ingestion & ETL) operating independently.
Uses the SemanticChunker to process raw text, preparing it for the Intelligence Layer.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.ingestion.semantic_chunker import SemanticChunker
from core.system.provenance_logger import ProvenanceLogger, ActivityType

def run_data_layer():
    print(">>> Initializing Data Layer (SemanticChunker)...")

    chunker = SemanticChunker()
    logger = ProvenanceLogger()

    # 1. Simulate Raw Data Ingestion (e.g., from SEC 10-K)
    print("[DataLayer] Ingesting raw document...")
    raw_document = """
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

    The following discussion and analysis of our financial condition and results of operations should be read in conjunction with our consolidated financial statements and the related notes included elsewhere in this Annual Report on Form 10-K.

    Overview
    We are a leading provider of technology solutions. Our revenue increased by 15% year-over-year due to strong demand for our cloud services.

    Liquidity and Capital Resources
    As of December 31, 2023, we had cash and cash equivalents of $5.2 billion. We believe our existing cash and cash equivalents will be sufficient to meet our working capital and capital expenditure needs for at least the next 12 months.
    """

    # 2. Process Data (ETL)
    print(f"[DataLayer] Processing document (Length: {len(raw_document)} chars)...")
    chunks = chunker.split_text(raw_document)

    print(f"[DataLayer] Generated {len(chunks)} semantic chunks.")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk[:50]}...\n")

    # 3. Log Provenance (Data Lineage)
    logger.log_activity(
        agent_id="SemanticIngestor",
        activity_type=ActivityType.INGESTION,
        input_data={"source": "10-K_Form_Item7", "length": len(raw_document)},
        output_data={"chunks_count": len(chunks), "chunk_sizes": [len(c) for c in chunks]},
        data_source="SEC_EDGAR_Simulated",
        capture_full_io=True
    )

    print(">>> Data Layer processing complete. Provenance logs generated.")

if __name__ == "__main__":
    run_data_layer()
