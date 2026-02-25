import sys
import os
import asyncio

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.system.memory_manager import VectorMemoryManager
from core.utils.logging_utils import SwarmLogger

def main():
    print("--- Repo Usage: Memory & Logging ---")

    # 1. System Logger
    logger = SwarmLogger()
    logger.log_event("EXAMPLE_RUN", "User", {"details": "Running repo_usage example"})
    print("Logged event to SwarmLogger.")

    # 2. Vector Memory (Mock check)
    try:
        memory = VectorMemoryManager()
        print("Initialized VectorMemoryManager.")

        # Simulating saving
        doc_id = "example_doc_001"
        content = "Adam v26.0 is a Neuro-Symbolic architecture."
        metadata = {"type": "documentation", "version": "26.0"}

        # Note: Actual saving requires a vector DB connection (Chroma/Pinecone)
        # We wrap in try/except to handle if DB is not running locally.
        try:
            # memory.save_document(doc_id, content, metadata)
            print("(Skipping actual DB write in example to avoid connection errors)")
        except Exception as e:
            print(f"DB Write Skipped: {e}")

    except Exception as e:
        print(f"Memory Init Failed (Expected if no DB): {e}")

if __name__ == "__main__":
    main()
