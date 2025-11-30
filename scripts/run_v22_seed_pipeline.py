import json
import sys
import os
import asyncio

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.v22_quantum_pipeline.quantum_source import QuantumMarketGenerator
from core.v22_quantum_pipeline.data_expander import expand_data
from core.v22_quantum_pipeline.async_loader import main as async_loader_main

def run_pipeline():
    print("--- Starting Adam v22.0 Quantum-Enhanced Pipeline ---")

    # 1. Load Seed File
    seed_file = 'adam_v22_seed.json'
    if not os.path.exists(seed_file):
        print(f"Error: {seed_file} not found.")
        return

    with open(seed_file, 'r') as f:
        seed_data = json.load(f)

    pipeline_artifacts = seed_data['adam_v22_seed_file']['pipeline_artifacts']

    # Step 1: Execute Module 1 (Quantum Generator)
    print("\n--- Step 1: Generating Latent Market Vectors (Quantum Source) ---")
    gen = QuantumMarketGenerator()
    # Generate some samples (e.g., 5)
    latent_tensors = gen(5)
    # Check shape
    shape = latent_tensors.shape if hasattr(latent_tensors, 'shape') else (len(latent_tensors), len(latent_tensors[0]))
    print(f"Generated Latent Tensor Shape: {shape}")

    # Step 2: Expand Module 2 Data
    print("\n--- Step 2: Expanding Training Data with Latent Tensors ---")
    module_2_payload = pipeline_artifacts['module_2_training_data']['data_payload']
    expanded_data = expand_data(module_2_payload, latent_tensors)
    print(f"Original Size: {len(module_2_payload)}, Expanded Size: {len(expanded_data)}")

    # Step 3: Save to JSONL
    output_file = 'adam_v22_instruction_tuning.jsonl'
    print(f"\n--- Step 3: Saving Augmented Data to {output_file} ---")
    with open(output_file, 'w') as f:
        for entry in expanded_data:
            json.dump(entry, f)
            f.write('\n')
    print("Data saved successfully.")

    # Step 4: Execute Module 3 (Async Loader)
    print("\n--- Step 4: Initiating Async LoRA Loading Protocol ---")
    # The async_loader_main expects to be run, but it might try to use asyncio.run itself if called directly.
    # However, I imported 'main' from async_loader which is an async function in my implementation?
    # Wait, in my async_loader.py, main is async: "async def main():".
    # And the block "if __name__ == '__main__': asyncio.run(main())".
    # So if I import main, I can just await it or run it.
    # But I am in a synchronous function run_pipeline. I should use asyncio.run(async_loader_main()).
    # Wait, if async_loader_main calls other async functions, it is fine.
    try:
        asyncio.run(async_loader_main())
    except Exception as e:
        print(f"Error during async loading: {e}")

    print("\n--- Adam v22.0 Pipeline Execution Complete ---")

if __name__ == '__main__':
    run_pipeline()
