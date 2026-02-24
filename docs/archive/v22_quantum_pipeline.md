# Adam v22.0 Quantum-Enhanced Pipeline

This document describes the implementation of the Adam v22.0 Quantum-Enhanced Generative AI Pipeline, bootstrapped from the `adam_v22_seed.json` file.

## Overview

The v22.0 pipeline integrates Quantum Variational Circuits (VQC) with the existing instruction tuning process to create "future-aligned" training data. This architecture ensures that the model is not just trained on historical data but also seeded with latent vectors representing potential market futures.

## Components

### 1. Seed File (`adam_v22_seed.json`)
The master configuration file that contains the source code and data for the pipeline. It serves as a self-contained portable specification for the v22.0 system.

### 2. Quantum Source (`core/v22_quantum_pipeline/quantum_source.py`)
- **Role:** The Generator.
- **Function:** Uses PennyLane and PyTorch to generate synthetic "Latent Market Vectors".
- **Fallback:** Includes a mock generator if quantum libraries are not available.

### 3. Data Expander (`core/v22_quantum_pipeline/data_expander.py`)
- **Role:** The Expander.
- **Function:** Takes the instruction tuning data and "augments" it by injecting the generated latent vectors into the system prompts. This creates multiple variations of each training example, each conditioned on a different "quantum state".

### 4. Async Loader (`core/v22_quantum_pipeline/async_loader.py`)
- **Role:** The Sink.
- **Function:** An asynchronous data loader designed to stream the augmented data into a training loop (e.g., LoRA fine-tuning) without blocking.

## Execution Protocol

To run the pipeline, execute the orchestration script:

```bash
python scripts/run_v22_seed_pipeline.py
```

This script performs the following steps:
1.  **Load Seed:** Reads `adam_v22_seed.json`.
2.  **Generate Tensors:** Runs the Quantum Source to get latent vectors.
3.  **Expand Data:** Combines the seed data with the vectors.
4.  **Save Data:** Outputs `adam_v22_instruction_tuning.jsonl`.
5.  **Verify Loading:** Runs the Async Loader to verify the data can be streamed.

## Requirements

- Python 3.8+
- `pennylane` (optional, mocked if missing)
- `torch` (optional, mocked if missing)
- `datasets` (optional, mocked if missing)
- `asyncio`
