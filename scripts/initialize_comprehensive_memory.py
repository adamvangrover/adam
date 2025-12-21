import logging
import os
import sys

# Ensure repo root is in path
sys.path.append(os.getcwd())

from core.learning.fine_tuning_driver import FineTuningDriver
from core.system.memory_consolidator import MemoryConsolidator

logging.basicConfig(level=logging.INFO)

def main():
    print("--- Initializing Comprehensive Memory System ---")

    # 1. Consolidate Memory (Repo + History + Financials)
    consolidator = MemoryConsolidator()
    consolidator.consolidate()

    manifest = consolidator.generate_system_manifest()
    print("\n" + manifest)

    os.makedirs("docs", exist_ok=True)
    with open("docs/SYSTEM_MANIFEST.md", "w") as f:
        f.write(manifest)
    print("System Manifest saved to docs/SYSTEM_MANIFEST.md")

    # 2. Generate Fine-Tuning Dataset
    print("\n--- Generating Fine-Tuning Dataset ---")
    driver = FineTuningDriver()
    driver.generate_dataset()
    print("Fine-Tuning Dataset generation complete.")

if __name__ == "__main__":
    main()
