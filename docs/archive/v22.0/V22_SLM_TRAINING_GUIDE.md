# Developer Guide: Training and Using the v22.0 SLM-LoRA Agent Brains

## 1. Introduction

This document provides the technical context for the "artisanal" datasets located in `data/artisanal_training_sets/`. As noted in the `Adam_v22.0_Portable_Config.json`, these datasets are not intended for direct use in Retrieval-Augmented Generation (RAG). Instead, they are high-quality, hand-crafted examples designed specifically for finetuning a suite of Small Language Models (SLMs) using Low-Rank Adaptation (LoRA).

The strategic goal of this "SLM-LoRA Agent Stack" is to create a set of highly specialized, computationally efficient, and reliable "expert tools". Each "brain" is trained to perform one specific, repetitive task and, critically, to output its analysis in a structured, machine-readable JSON format. The main v22.0 LLM simulates "calling" these tools and uses their structured JSON output as the factual basis for its grounded analysis and provenance citations.

## 2. The "Brains" and Their Purpose

| Agent Brain ID | Artisanal Dataset | Core Purpose |
| :--- | :--- | :--- |
| `SNC-Analyst-Brain-v1.0` | `artisanal_data_snc_v1.jsonl` | Automates mandatory regulatory credit analysis (SNC rating). |
| `Red-Team-Brain-v1.0` | `artisanal_data_redteam_v1.jsonl` | Programmatically challenges a baseline analysis by identifying unstated assumptions. |
| `HouseView-Macro-Brain-v1.0` | `artisanal_data_houseview_v1.jsonl` | Serves as the single source of truth for the firm's macroeconomic opinions. |
| `Behavioral-Economics-Brain-v1.0` | `artisanal_data_behavioral_v1.jsonl` | Translates qualitative behavioral finance concepts into quantitative risk parameters. |

## 3. Training Methodology (SLM-LoRA)

The recommended methodology for training these brains is Supervised Fine-Tuning (SFT) using a LoRA-enabled training script.

### Recommended Base Models:

- **meta-llama/Llama-3.1-8B-Instruct** (or similar high-performing 8B parameter model) is an ideal base. Its size makes it efficient to finetune and serve, while its instruction-following capabilities are excellent for adhering to the strict JSON output format.

### Training Steps:

1.  **Environment Setup:** Use a standard Python environment with libraries such as `transformers`, `peft` (for LoRA), `accelerate`, and `torch`.
2.  **Load Dataset:** Load the target `.jsonl` file from `data/artisanal_training_sets/`. Each line is a complete JSON object with "prompt" and "completion" keys.
3.  **Format Data:** The prompt and completion must be formatted into a single string that matches the base model's chat template. For example:
    ```
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    {prompt_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {completion_content}<|eot_id|>
    ```
4.  **Configure LoRA:** Use the `peft` library to create a `LoraConfig`. Target the attention blocks of the base model (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`). A low rank (e.g., `r=16` or `r=32`) is typically sufficient.
5.  **Instantiate Trainer:** Use the `transformers.SFTTrainer`. Pass the configured model, dataset, and LoRA config to the trainer.
6.  **Run Training:** A small number of epochs (e.g., 3-5) is usually enough for convergence on these specialized tasks.
7.  **Save Adapter:** Save the trained LoRA adapter. This small set of weights (the "brain") can then be dynamically loaded alongside the base SLM for inference.

## 4. Inference and Usage

At inference time, the main v22.0 LLM does not *actually* call these models. It *narrates* the process. A separate orchestration layer is responsible for:
1.  Identifying the need for a specialized tool based on the v22.0 LLM's narration (e.g., it says "Invoking Red Team Agent...").
2.  Loading the base SLM and attaching the appropriate LoRA adapter (e.g., `Red-Team-Brain-v1.0`).
3.  Executing the prompt against the adapted SLM to get the structured JSON output.
4.  Feeding this JSON output back into the main v22.0 LLM's context so it can be used for the critical "Groundedness & Provenance" step of its operational loop.
