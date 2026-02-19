# Adam SLM Pipeline: Protocol "DeepMind Milestone"

> **"Rigorous measurement precedes improvement."**

This document defines the operational protocol for the `tinker_lab/pipeline_v2/` environment. It is a specialized extension of the global `AGENTS.md` tailored for high-stakes Foundation Model training and alignment.

## 1. The DeepMind Milestone Protocol

All training runs must adhere to a strict lifecycle of "Milestones". We do not just "run scripts"; we advance through gates.

### The Gates
1.  **GATE 0: HYPOTHESIS & CONFIG (Planning)**
    *   State the intent: "Fine-tune Adam-SLM-Alpha on Artisanal Batch 001 to improve financial reasoning."
    *   Artifact: `config.yaml` locked.
2.  **GATE 1: DATA INTEGRITY (Validation)**
    *   Verify input data distribution, weights, and formatting.
    *   Artifact: `data_validation_report.json`.
3.  **GATE 2: TRAINING DYNAMICS (Execution)**
    *   Execute the training loop (Tinker SDK).
    *   Monitor loss curves and resource usage.
    *   Artifact: `training_logs.jsonl`.
4.  **GATE 3: ALIGNMENT & EVAL (Reflection)**
    *   Run automated evals (OAI-aligned metrics).
    *   Check for regression in core capabilities.
    *   Artifact: `eval_scorecard.md`.
5.  **GATE 4: DEPLOYMENT (Release)**
    *   Update `seed_prompts.json` and weight manifests.
    *   Artifact: `adapter_weights_manifest.json`.

## 2. System 2 Logging

Agents operating in this pipeline must log their "thought process" not just their output.
*   **Use:** `milestone_tracker.py`
*   **Format:**
    ```markdown
    ## [2025-10-27 14:00] GATE 2 START
    **Context:** Training job initiated.
    **Reasoning:** Data validation passed (Score: 1.0). Resource check confirmed. Proceeding to optimize.
    ```

## 3. Environment Awareness

The pipeline runs in two modes:
*   **`MODE: MOCK` (Default):** Simulates training steps for CI/CD and development. Uses synthetic delays and outputs.
*   **`MODE: LIVE` (Restricted):** Connects to the Tinker remote cluster. Requires `TINKER_API_KEY`.

**Directive:** "Assume MOCK unless explicitly authorized."
