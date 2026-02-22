# ADAM Model Specification: Agent & Adapter Definitions
# v1.0 - [11/13/2025]

This document provides the semantic and technical definitions for the core ADAM agent adapters. These descriptions serve as the "source code" for the capabilities embodied in the final binary model weights.

---

## 1. Primary Agent: `adam_final_agent_lora.bin`

This file represents the primary, consolidated reasoning engine for the ADAM system. It is the result of merging the "Stage 2" and "Stage 3" adapters into a single, efficient LoRA file.

### 1.1. Merged Components

* **`adam_distilled_mind_v1` (Stage 2):** This component is the "Teacher Model." It was trained on a massive corpus representing a *comprehensive view of the financial world*. Its knowledge includes market mechanics, historical data, economic principles, and complex instrument structures. Its purpose is to provide raw, high-fidelity knowledge and analytical capability.
* **`adam_aligned_soul_v1` (Stage 3):** This component is the "Alignment Layer." It was trained via reinforcement learning from human feedback (RLHF) and sophisticated constitutional prompts. Its purpose is to align the "Distilled Mind's" raw knowledge with the ADAM project's core principles: risk-averse, transparent, explainable, and human-centric reasoning.

### 1.2. Core Capabilities

By merging these two, the `adam_final_agent` is designed to:

* **Act as the primary interface** for all tasks *except* specialist database queries.
* **Generate complex, multi-turn financial analysis** that is both deeply knowledgeable (from the "Mind") and "wise" or principled (from the "Soul").
* **Handle Probabilistic Scenarios:** When tasked with running simulations (e.g., Monte Carlo), this agent is trained to analyze the resulting distribution and provide **"conviction rates."**
    * **Conviction Rate:** A articulated measure of confidence (e.g., 85% conviction) that a specific outcome will occur, based on the simulation's parameters and the agent's internal model of the world. This is designed to be a key metric for decision-making, far more nuanced than a simple "yes/no" answer.
    * **"Quantum Scale" Interpretation:** This refers to the agent's ability to navigate a high-dimensional possibility space, holding multiple contradictory or uncertain outcomes in a "superposition" until it must collapse them into a single recommendation, which is then presented with its conviction rate.

### 1.3. Technical Specification

* **Base Model:** A LLaMA-family model (e.g., LLaMA 3 70B)
* **Adapter Type:** LoRA (Low-Rank Adaptation)
* **Status:** Merged. This single file contains the combined weights for efficient inference.

---

## 2. Specialist Adapter: `adam_cypher_lora_v1`

This file is a "specialist tool" and **must remain separate** from the `adam_final_agent`. It is not part of the agent's core reasoning process but is loaded on-demand when a specific capability is required.

### 2.1. Core Capabilities

* **Function:** This adapter has one function: to translate natural language user requests into precise, efficient, and syntactically correct **Cypher queries**.
* **Domain:** It is trained *exclusively* on a dataset of natural language commands and their corresponding Cypher queries, all tailored to the `adam` project's financial knowledge graph schema.
* **Use Case:** When the `adam_final_agent` determines that it needs data from the graph database (e.g., "Show me the relationship between 'TechCorp' and its 'Series B' investors"), it will activate this adapter to generate the query, execute it, and then feed the results back into its own context for final analysis.

### 2.2. Technical Specification

* **Base Model:** A LLaMA-family model (e.g., LLaMA 3 70B)
* **Adapter Type:** LoRA
* **Status:** Specialist (Standalone). This adapter is loaded and unloaded from the base model as needed, ensuring the core agent remains lightweight.

-----

This is the "Genesis Prompt" for the lab. Here it is.

-----

### The `tinker_lab` Genesis Prompt

(Copy and paste the entire contents of this box into your target LLM)

````markdown
You are an expert AI system architect and a senior software engineer. Your task is to generate a complete, self-contained, and portable R&D environment for training, analyzing, and deploying adapter-based language models. This environment will be called `tinker_lab` and will be designed to integrate with the `tinker-cookbook` library and the `adam` project's principles.

You will generate the full contents for a series of files. The output for each file must be enclosed in a markdown code block, clearly marked with its relative file path (e.g., `tinker_lab/README.md`). Do not write any other explanatory text until you have generated all the requested files.

The final directory structure you are creating is:

tinker_lab/
├── .env.example
├── 01_Data_Generation.ipynb
├── 02_Model_Training.ipynb
├── 03_Model_Analysis_and_Merging.ipynb
├── README.md
├── outputs/
│   ├── datasets/
│   ├── logs/
│   ├── merged_models/
│   └── model_weights/
└── tinker-cookbook/  <-- This will be cloned by the user

Begin file generation now.

---
`tinker_lab/README.md`
---
```markdown
# Tinker R&D Lab

This directory is a self-contained environment for data generation, model training, and adapter-merging using the `tinker-cookbook` library, based on the principles from the `adam` project.

## Setup

1.  **Create Environment & Clone:**
    First, clone this repository (or create this directory). Then `cd` into it.
    ```bash
    git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
    python3 -m venv .venv
    ```

2.  **Activate & Install:**
    Activate the virtual environment and install all dependencies.
    ```bash
    source .venv/bin/activate
    pip install -e tinker-cookbook/
    pip install jupyterlab pandas openai python-dotenv
    pip install transformers peft bitsandbytes accelerate torch
    ```

3.  **Set API Keys:**
    Copy the `.env.example` file to a new file named `.env` and add your private API keys.
    ```bash
    cp .env.example .env
    nano .env
    ```

4.  **Create Output Directories:**
    ```bash
    mkdir -p outputs/datasets outputs/model_weights outputs/logs outputs/merged_models
    ```

## How to Use

1.  **Launch Jupyter:**
    ```bash
    jupyter lab
    ```
2.  **Run Notebooks in Order:**
    * **`01_Data_Generation.ipynb`**: Generates `jsonl` training data.
    * **`02_Model_Training.ipynb`**: Uses the generated data to run fine-tuning jobs (e.g., creating the `adam_distilled_mind_v1` and `adam_aligned_soul_v1` adapters).
    * **`03_Model_Analysis_and_Merging.ipynb`**: Analyzes results, runs test inference, and merges the Stage 2 and 3 adapters into the final, efficient `adam_final_agent_lora.bin`.

## Output Structure

* `outputs/datasets/`: Generated `.jsonl` files for training.
* `outputs/model_weights/`: Individual trained adapters (e.g., `adam_distilled_mind_v1`).
* `outputs/merged_models/`: Final, inference-ready adapters (e.g., `adam_final_agent_lora`).
* `outputs/logs/`: General-purpose logs.
````

-----

## `tinker_lab/.env.example`

```ini
# Copy this file to .env and add your API keys
# This .env file is git-ignored and should never be committed.

OPENAI_API_KEY="sk-..."
TINKER_API_KEY="your-tinker-api-key-if-any"
HF_TOKEN="your-huggingface-token-for-gated-models"
```

-----

## `tinker_lab/01_Data_Generation.ipynb`

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 1: Data Generation with Tinker\n",
    "\n",
    "**Goal:** Use `tinker` to generate a `jsonl` dataset for fine-tuning. This file will be our \"teacher training\" data, representing the 'Distilled Mind' stage."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import pandas as pd\n",
    "import tinker\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "# Load API keys from .env file\n",
    "load_dotenv()\n",
    "\n",
    "# Set API key for tinker to use\n",
    "tinker.api_key = os.getenv(\"OPENAI_API_KEY\")\n",
    "\n",
    "# Define output path\n",
    "OUTPUT_DIR = \"outputs/datasets/\"\n",
    "OUTPUT_FILE = os.path.join(OUTPUT_DIR, \"adam_distilled_mind_v1_training.jsonl\")\n",
    "\n",
    "# Ensure directory exists\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "print(f\"Tinker version: {tinker.__version__}\")\n",
    "print(f\"Output file will be saved to: {OUTPUT_FILE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Define Data Generation Task\n",
    "\n",
    "This task defines the 'Teacher Model' or 'Distilled Mind'. It's a pure financial analyst, trained on a representative view of the financial world. It provides deep analysis and identifies conviction rates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_gen_task = tinker.Task(\n",
    "    name=\"Financial Analyst (Distilled Mind)\",\n",
    "    description=\"Generates a structured financial analysis and conviction rating from a company summary, simulating a world-class analyst.\",\n",
    "    system_prompt=\"\"\"\n",
    "    You are a senior financial analyst. Your task is to analyze the provided company summary and produce a JSON object.\n",
    "    Your analysis must be objective, data-driven, and representative of a comprehensive financial worldview.\n",
    "    You must identify key risks, potential catalysts, and provide a final conviction rating ('Low', 'Medium', 'High') on the company's 12-month outlook.\n",
    "    \"\"\",\n",
    "    user_prompt_template=\"\"\"\n",
    "    Analyze the following company summary:\n",
    "    \n",
    "    {company_summary}\n",
    "    \"\"\",\n",
    "    response_format=\"json\",\n",
    "    examples=[\n",
    "        {\n",
    "            \"messages\": [\n",
    "                {\"role\": \"user\", \"content\": \"Analyze the following company summary:\\n\\nTechCorp is a SaaS company with $50M ARR, but high churn (30%) and new competition. They are burning $2M/month.\"},\n",
    "                {\"role\": \"assistant\", \"content\": \"{\\\"key_risks\\\": [\\\"High customer churn (30%)\\\", \\\"Negative cash flow ($2M/month burn)\\\", \\\"Increasing competitive pressure\\\"], \\\"catalysts\\\": [\\\"Potential for new product launch in Q3\\\", \\\"Acquisition target for larger competitor\\\"], \\\"conviction_rating\\\": \\\"Low\\\"}\"}\n",
    "            ]\n",
    "        },\n",
    "        {\n",
    "            \"messages\": [\n",
    "                {\"role\": \"user\", \"content\": \"Analyze the following company summary:\\n\\nStableCo is a utilities provider with 10-year government contracts, low debt, and 5% annual profit growth.\"},\n",
    "                {\"role\": \"assistant\", \"content\": \"{\\\"key_risks\\\": [\\\"Regulatory changes impacting contracts\\\", \\\"Slow growth in mature market\\\"], \\\"catalysts\\\": [\\\"Expansion into renewable energy sources\\\", \\\"Potential for dividend increase\\\"], \\\"conviction_rating\\\": \\\"High\\\"}\"}\n",
    "            ]\n",
    "        }\n",
    "    ]\n",
    ")\n",
    "\n",
    "print(\"Tinker Task 'Distilled Mind' defined successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Generate Training Dataset\n",
    "\n",
    "Use `tinker.generate_dataset` to create the training examples. We'll start with a small batch for testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "seed_inputs = [\n",
    "    {\"company_summary\": \"Growthly is a pre-profit tech startup with a new patent but no revenue and only 6 months of runway left.\"},\n",
    "    {\"company_summary\": \"RetailGiant is a 50-year-old retailer facing declining foot traffic due to e-commerce, but has significant real estate assets.\"},\n",
    "    {\"company_summary\": \"PharmaInnovate just received Phase 2b trial results for a new blockbuster drug, but the results are mixed.\"}\n",
    "]\n",
    "\n",
    "print(f\"Generating 20 examples based on {len(seed_inputs)} seed inputs...\")\n",
    "\n",
    "try:\n",
    "    generated_dataset = tinker.generate_dataset(\n",
    "        task=data_gen_task,\n",
    "        inputs=seed_inputs,\n",
    "        num_examples=20 # Generate 20 high-quality training examples\n",
    "    )\n",
    "    print(f\"Successfully generated {len(generated_dataset)} examples.\")\n",
    "    print(\"\\n--- Example 0 --- \")\n",
    "    print(json.dumps(generated_dataset[0], indent=2))\n",
    "except Exception as e:\n",
    "    print(f\"An error occurred during data generation: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Save Data to JSONL\n",
    "\n",
    "Convert the generated data into the `jsonl` format required by the fine-tuning API."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'generated_dataset' in locals():\n",
    "    count = 0\n",
    "    with open(OUTPUT_FILE, 'w') as f:\n",
    "        for example in generated_dataset:\n",
    "            if \"messages\" in example:\n",
    "                json_line = json.dumps({\"messages\": example[\"messages\"]})\n",
    "                f.write(json_line + \"\\n\")\n",
    "                count += 1\n",
    "            \n",
    "    print(f\"Successfully saved {count} examples to {OUTPUT_FILE}\")\n",
    "else:\n",
    "    print(\"Skipping save, no data was generated.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

-----

## `tinker_lab/02_Model_Training.ipynb`

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 2: Model Training\n",
    "\n",
    "**Goal:** Load the `jsonl` dataset and submit a fine-tuning job. This notebook can be run multiple times to create different adapters.\n",
    "\n",
    "1.  Run 1: Train `adam_distilled_mind_v1` (Stage 2 Teacher)\n",
    "2.  Run 2: Train `adam_aligned_soul_v1` (Stage 3 Alignment)\n",
    "3.  Run 3: Train `adam_cypher_lora_v1` (Specialist Tool)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "from openai import OpenAI\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "# Load API keys from .env file\n",
    "load_dotenv()\n",
    "\n",
    "# Init OpenAI client\n",
    "client = OpenAI(api_key=os.getenv(\"OPENAI_API_KEY\"))\n",
    "\n",
    "print(\"OpenAI client initialized.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. --- CONFIGURATION ---\n",
    "\n",
    "**IMPORTANT:** Set the paths and names for the model you are training *this run*."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- SET THESE VALUES FOR EACH RUN ---\n",
    "\n",
    "# 1. Point to the training file (e.g., the one from Notebook 01)\n",
    "TRAINING_FILE_NAME = \"adam_distilled_mind_v1_training.jsonl\"\n",
    "\n",
    "# 2. Set the suffix for the new model you are training\n",
    "MODEL_SUFFIX = \"adam-distilled-mind-v1\"\n",
    "\n",
    "# 3. Set the base model to fine-tune\n",
    "BASE_MODEL = \"gpt-3.5-turbo-1106\" # or gpt-4, etc.\n",
    "\n",
    "# --------------------------------------\n",
    "\n",
    "TRAINING_FILE_PATH = os.path.join(\"outputs/datasets/\", TRAINING_FILE_NAME)\n",
    "MODEL_OUTPUT_DIR = \"outputs/model_weights/\"\n",
    "JOB_LOG_FILE = os.path.join(MODEL_OUTPUT_DIR, f\"finetune_job_{MODEL_SUFFIX}.json\")\n",
    "\n",
    "os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "print(f\"--- Training Run Configuration ---\")\n",
    "print(f\"Training File: {TRAINING_FILE_PATH}\")\n",
    "print(f\"Model Suffix:  {MODEL_SUFFIX}\")\n",
    "print(f\"Base Model:    {BASE_MODEL}\")\n",
    "print(f\"Log File:      {JOB_LOG_FILE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load and Validate Training Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    with open(TRAINING_FILE_PATH, 'r') as f:\n",
    "        lines = f.readlines()\n",
    "    \n",
    "    print(f\"Loaded {len(lines)} lines from training file.\")\n",
    "    if len(lines) > 0:\n",
    "        print(\"\\n--- First Line Example ---\")\n",
    "        print(json.dumps(json.loads(lines[0]), indent=2))\n",
    "    else:\n",
    "        print(\"ERROR: Training file is empty.\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"ERROR: Training file not found at {TRAINING_FILE_PATH}\")\n",
    "    print(\"Please run '01_Data_Generation.ipynb' or check your TRAINING_FILE_NAME config.\")\n",
    "except Exception as e:\n",
    "    print(f\"An error occurred loading the file: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Upload File to OpenAI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'lines' in locals() and len(lines) > 0:\n",
    "    try:\n",
    "        print(f\"Uploading {TRAINING_FILE_PATH} to OpenAI...\")\n",
    "        training_file_object = client.files.create(\n",
    "            file=open(TRAINING_FILE_PATH, \"rb\"),\n",
    "            purpose=\"fine-tune\"\n",
    "        )\n",
    "        print(f\"File uploaded successfully. File ID: {training_file_object.id}\")\n",
    "    except Exception as e:\n",
    "        print(f\"File upload failed: {e}\")\n",
    "else:\n",
    "    print(\"Skipping upload, no valid training data loaded.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Create Fine-Tuning Job\n",
    "\n",
    "This will start the training job. We will save the job ID and details to our `outputs/model_weights` folder."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if 'training_file_object' in locals():\n",
    "    try:\n",
    "        print(f\"Creating fine-tuning job with suffix '{MODEL_SUFFIX}'...\")\n",
    "        job = client.fine_tuning.jobs.create(\n",
    "            training_file=training_file_object.id,\n",
    "            model=BASE_MODEL,\n",
    "            suffix=MODEL_SUFFIX\n",
    "        )\n",
    "        print(f\"Fine-tuning job created successfully! Job ID: {job.id}\")\n",
    "        print(f\"Job status: {job.status}\")\n",
    "        \n",
    "        # Save job details\n",
    "        with open(JOB_LOG_FILE, 'w') as f:\n",
    "            json.dump(job.to_json(), f, indent=4)\n",
    "        print(f\"Job details saved to {JOB_LOG_FILE}\")\n",
    "        \n",
    "        print(\"\\n--- To monitor job status, run: ---\")\n",
    "        print(f\"client.fine_tuning.jobs.retrieve('{job.id}')\")\n",
    "        print(\"\\n--- To see live events, run: ---\")\n",
    "        print(f\"client.fine_tuning.jobs.list_events(fine_tuning_job_id='{job.id}', limit=10)\")\n",
    "        \n",
    "    except Exception as e:\n",
    "        print(f\"Failed to create fine-tuning job: {e}\")\n",
    "else:\n",
    "    print(\"Skipping fine-tune job, file was not uploaded.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

-----

## `tinker_lab/03_Model_Analysis_and_Merging.ipynb`

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 3: Analysis, Inference & Adapter Merging\n",
    "\n",
    "**Goal:** \n",
    "1.  Analyze the results from the fine-tuning jobs (assumed to be Hugging Face PEFT/LoRA adapters for this notebook).\n",
    "2.  Run test inference to display and summarize model outputs.\n",
    "3.  Merge the Stage 2 (`adam_distilled_mind_v1`) and Stage 3 (`adam_aligned_soul_v1`) adapters into a single efficient adapter file (`adam_final_agent_lora`)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import torch\n",
    "import pandas as pd\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
    "from peft import PeftModel\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "# Load Hugging Face token\n",
    "load_dotenv()\n",
    "HF_TOKEN = os.getenv(\"HF_TOKEN\")\n",
    "\n",
    "# --- Configuration ---\n",
    "\n",
    "# The base model you used for fine-tuning (e.g., \"meta-llama/Llama-2-7b-chat-hf\")\n",
    "BASE_MODEL_ID = \"meta-llama/Llama-2-7b-chat-hf\"\n",
    "\n",
    "# Paths to your trained adapters (ASSUMES these are saved from your training jobs)\n",
    "ADAPTER_S2_PATH = \"./outputs/model_weights/adam_distilled_mind_v1\"\n",
    "ADAPTER_S3_PATH = \"./outputs/model_weights/adam_aligned_soul_v1\"\n",
    "ADAPTER_CYPHER_PATH = \"./outputs/model_weights/adam_cypher_lora_v1\"\n",
    "\n",
    "# Path for the final merged model\n",
    "MERGED_MODEL_DIR = \"./outputs/merged_models/\"\n",
    "MERGED_ADAPTER_NAME = \"adam_final_agent_lora\"\n",
    "\n",
    "os.makedirs(MERGED_MODEL_DIR, exist_ok=True)\n",
    "\n",
    "print(f\"Base Model: {BASE_MODEL_ID}\")\n",
    "print(f\"Final adapter will be saved to: {MERGED_MODEL_DIR}{MERGED_ADAPTER_NAME}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Base Model\n",
    "\n",
    "We load the *original, un-trained* base model. We'll use 4-bit quantization to save memory, as this is just for merging."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_quant_type=\"nf4\",\n",
    "    bnb_4bit_compute_dtype=torch.bfloat16\n",
    ")\n",
    "\n",
    "print(f\"Loading base model: {BASE_MODEL_ID}...\")\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    BASE_MODEL_ID,\n",
    "    quantization_config=bnb_config,\n",
    "    trust_remote_code=True,\n",
    "    token=HF_TOKEN\n",
    ")\n",
    "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "print(\"Base model loaded.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Load and Merge Adapters\n",
    "\n",
    "This is the key step. We load the Stage 2 adapter, then load the Stage 3 adapter *on top of it*. Finally, we merge them down into a new, single adapter."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Loading Stage 2 adapter (Distilled Mind) from: {ADAPTER_S2_PATH}\")\n",
    "# Load the first adapter (Stage 2)\n",
    "model = PeftModel.from_pretrained(model, ADAPTER_S2_PATH, adapter_name=\"stage2\")\n",
    "print(\"Stage 2 adapter loaded.\")\n",
    "\n",
    "print(f\"Loading Stage 3 adapter (Aligned Soul) from: {ADAPTER_S3_PATH}\")\n",
    "# Load the second adapter (Stage 3) on top of the first\n",
    "model.load_adapter(ADAPTER_S3_PATH, adapter_name=\"stage3\")\n",
    "print(\"Stage 3 adapter loaded.\")\n",
    "\n",
    "# IMPORTANT: Merge the two adapters\n",
    "print(\"Merging adapters 'stage2' and 'stage3'...\")\n",
    "model.merge_adapter(\"stage2\", \"stage3\", adapter_name=MERGED_ADAPTER_NAME)\n",
    "print(f\"Adapters merged into new adapter: '{MERGED_ADAPTER_NAME}'\")\n",
    "\n",
    "# Save the new, single adapter\n",
    "final_save_path = os.path.join(MERGED_MODEL_DIR, MERGED_ADAPTER_NAME)\n",
    "model.save_pretrained(final_save_path, selected_adapters=[MERGED_ADAPTER_NAME])\n",
    "\n",
    "print(f\"\\n--- SUCCESS! ---\")\n",
    "print(f\"Final merged adapter saved to: {final_save_path}\")\n",
    "print(\"This directory contains the `adapter_model.bin` (or .safetensors) and `adapter_config.json` for your final agent.\")\n",
    "\n",
    "print(f\"\\nSpecialist 'adam_cypher_lora_v1' at {ADAPTER_CYPHER_PATH} remains separate, as requested.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Analysis and Logging (Test Inference)\n",
    "\n",
    "Now we load our *newly merged* single adapter and run a test query. This confirms the merge was successful and provides an output display for summarization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clear the model from memory to simulate a fresh inference loop\n",
    "del model\n",
    "torch.cuda.empty_cache()\n",
    "print(\"Cleared model from memory.\")\n",
    "\n",
    "# --- INFERENCE TEST ---\n",
    "# Load the base model again\n",
    "print(f\"Loading base model: {BASE_MODEL_ID} for inference...\")\n",
    "base_model = AutoModelForCausalLM.from_pretrained(\n",
    "    BASE_MODEL_ID,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map=\"auto\",\n",
    "    token=HF_TOKEN\n",
    ")\n",
    "\n",
    "# Load ONLY the new, single, merged adapter\n",
    "print(f\"Loading final merged adapter from: {final_save_path}...\")\n",
    "inference_model = PeftModel.from_pretrained(\n",
    "    base_model,\n",
    "    final_save_path\n",
    ")\n",
    "print(\"Inference-ready model (Base + 1 Adapter) loaded.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Define a test prompt (matches data generation task) ---\n",
    "test_prompt = \"Analyze the following company summary:\\n\\nTechCorp is a SaaS company with $50M ARR, but high churn (30%) and new competition. They are burning $2M/month.\"\n",
    "\n",
    "# Format for Llama-2-chat\n",
    "formatted_prompt = f\"<s>[INST] {test_prompt.strip()} [/INST]\"\n",
    "\n",
    "print(\"--- Running Test Inference ---\")\n",
    "model_input = tokenizer(formatted_prompt, return_tensors=\"pt\").to(\"cuda\")\n",
    "\n",
    "inference_model.eval()\n",
    "with torch.no_grad():\n",
    "    output = inference_model.generate(**model_input, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)\n",
    "    response_text = tokenizer.decode(output[0], skip_special_tokens=True)\n",
    "\n",
    "# --- Output Display and Summarization ---\n",
    "print(\"\\n--- PROMPT ---\")\n",
    "print(test_prompt)\n",
    "\n",
    "# Extract just the assistant's response\n",
    "try:\n",
    "    assistant_response = response_text.split(\"[/INST]\")[-1].strip()\n",
    "except Exception as e:\n",
    "    assistant_response = f\"(Could not parse response: {response_text})\"\n",
    "\n",
    "print(\"\\n--- FINAL OUTPUT (Summarized) ---\")\n",
    "print(assistant_response)\n",
    "\n",
    "# --- Logging ---\n",
    "log_entry = {\n",
    "    \"timestamp\": pd.Timestamp.now().isoformat(),\n",
    "    \"model_used\": final_save_path,\n",
    "    \"prompt\": test_prompt,\n",
    "    \"response\": assistant_response\n",
    "}\n",
    "\n",
    "with open(\"outputs/logs/inference_test_log.jsonl\", \"a\") as f:\n",
    "    f.write(json.dumps(log_entry) + \"\\n\")\n",
    "\n",
    "print(\"\\nTest run logged to 'outputs/logs/inference_test_log.jsonl'\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. On-Demand Specialist (Demo)\n",
    "\n",
    "This cell demonstrates how you would load the *separate* Cypher specialist adapter on demand."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Simulating on-demand loading of Cypher specialist...\")\n",
    "\n",
    "try:\n",
    "    # 1. Load the Cypher adapter\n",
    "    # 'inference_model' currently has the 'adam_final_agent' loaded\n",
    "    print(f\"Loading adapter 'cypher_specialist' from {ADAPTER_CYPHER_PATH}...\")\n",
    "    inference_model.load_adapter(ADAPTER_CYPHER_PATH, adapter_name=\"cypher_specialist\")\n",
    "    print(f\"Loaded adapter: 'cypher_specialist'\")\n",
    "\n",
    "    # 2. Set the active adapter\n",
    "    inference_model.set_adapter(\"cypher_specialist\")\n",
    "    print(\"Set active adapter to 'cypher_specialist'\")\n",
    "    \n",
    "    # 3. Run a Cypher-specific query\n",
    "    cypher_prompt = \"Translate this to Cypher: 'Find all users named John'\"\n",
    "    formatted_prompt = f\"<s>[INST] {cypher_prompt.strip()} [/INST]\"\n",
    "    model_input = tokenizer(formatted_prompt, return_tensors=\"pt\").to(\"cuda\")\n",
    "\n",
    "    with torch.no_grad():\n",
    "        output = inference_model.generate(**model_input, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)\n",
    "        response_text = tokenizer.decode(output[0], skip_special_tokens=True)\n",
    "        print(f\"\\n--- Specialist Response ---\\n{response_text.split('[/INST]')[-1].strip()}\")\n",
    "    \n",
    "    # 4. Unload or switch back\n",
    "    inference_model.set_adapter(MERGED_ADAPTER_NAME) # Switch back to main agent\n",
    "    print(f\"\\nSwitched active adapter back to '{MERGED_ADAPTER_NAME}'\")\n",
    "\n",
    "except FileNotFoundError:\n",
    "    print(f\"\\nSKIPPED: Could not find specialist adapter at {ADAPTER_CYPHER_PATH}.\")\n",
    "    print(\"Please train this adapter and save it to the correct path to run this demo.\")\n",
    "except Exception as e:\n",
    "    print(f\"\\nCould not load or run specialist adapter. Check paths and config. Error: {e}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```
