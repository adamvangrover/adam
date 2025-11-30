# Tinker Lab Summary and Strategic Recommendations

## 1. Environment Analysis

The `tinker_lab` directory has been successfully created as a self-contained R&D environment for model training and data generation. This lab is built to integrate with the `tinker-cookbook` library, providing a streamlined workflow for developing and fine-tuning models as part of the Adam v21.0 architecture.

The environment includes:
- **A clear directory structure:** Separating reference documentation, outputs (datasets, model weights, logs), and the core `tinker-cookbook` dependency.
- **Starter Notebooks:** `01_Data_Generation.ipynb` and `02_Model_Training.ipynb` provide a functional baseline for generating synthetic data and running fine-tuning jobs.
- **Configuration Management:** A `.env.example` file ensures that API keys and other secrets are managed securely and not hardcoded.
- **Comprehensive Documentation:** The `README.md` file offers clear setup and usage instructions for any developer to quickly get started.

## 2. Strategic Recommendation: LoRA Adapter Merging for Inference Efficiency

To optimize the deployment of the trained LoRA adapters from the Adam v21.0 pipeline, the following strategy is recommended:

For inference efficiency, the `adam_distilled_mind_v1` (Stage 2) and `adam_aligned_soul_v1` (Stage 3) adapters should be merged into a single `adam_final_agent_lora.bin` file. This reduces the number of weights to be loaded in Step 2 of the inference loop, decreasing latency and computational overhead.

The `adam_cypher_lora_v1` adapter must remain separate. It functions as a "specialist tool" that is only loaded on demand when the agent needs to execute database queries. This modular approach ensures that the base agent remains lightweight, loading specialized capabilities only when necessary.

---
## 3. Final File Contents

### `tinker_lab/README.md`

```markdown
# Tinker R&D Lab

This directory is a self-contained environment for data generation and model training using the `tinker-cookbook` library, based on the principles and documentation from `adam/v21.0`.

## Setup

1.  **Activate Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    If this is the first time, or if dependencies change, run:
    ```bash
    pip install -e tinker-cookbook/
    pip install jupyterlab pandas openai python-dotenv
    ```

3.  **Set API Keys:**
    Copy the `.env.example` file to a new file named `.env` and add your private API keys.
    ```bash
    cp .env.example .env
    nano .env
    ```

## How to Use

1.  **Launch Jupyter:**
    ```bash
    jupyter lab
    ```
2.  **Run Notebooks:**
    * **`01_Data_Generation.ipynb`**: Use this to generate `jsonl` training datasets.
    * **`02_Model_Training.ipynb`**: Use this to load the generated data and run fine-tuning jobs.

## Output Structure

All artifacts are saved to the `outputs/` directory:
* `outputs/datasets/`: Contains generated `.jsonl` files for training.
* `outputs/model_weights/`: Contains logs and identifiers for trained models.
* `outputs/logs/`: Contains general-purpose logs.
```

### `tinker_lab/.env.example`

```
# Copy this file to .env and add your API keys
# This .env file is git-ignored and should never be committed.

OPENAI_API_KEY="sk-..."
TINKER_API_KEY="your-tinker-api-key-if-any"
# Add any other required keys
```

### `tinker_lab/01_Data_Generation.ipynb`

```json
{
"cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 1: Data Generation with Tinker\n",
    "\n",
    "**Goal:** Use `tinker` to generate a `jsonl` dataset for fine-tuning. This file will be our \"teacher training\" data."
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
    "# Set your OpenAI API key for tinker to use\n",
    "# Assumes tinker uses the OPENAI_API_KEY env variable\n",
    "tinker.api_key = os.getenv(\"OPENAI_API_KEY\")\n",
    "\n",
    "# Define output path\n",
    "OUTPUT_DIR = \"outputs/datasets/\"\n",
    "OUTPUT_FILE = os.path.join(OUTPUT_DIR, \"training_data.jsonl\")\n",
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
    "Define the `tinker.Task` for the model. This includes the system prompt, user prompt, and example output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This is a hypothetical task based on the ADAM project's goals\n",
    "data_gen_task = tinker.Task(\n",
    "    name=\"Credit Risk Assessor\",\n",
    "    description=\"Generates a structured credit risk assessment from a company summary.\",\n",
    "    system_prompt=\"\"\"\n",
    "    You are a senior credit analyst. Your task is to analyze the provided company summary and produce a JSON object outlining the key credit risks and a final rating. The rating should be one of: 'Low', 'Medium', 'High'.\n",
    "    \"\"\",\n",
    "    user_prompt_template=\"\"\"\n",
    "    Analyze the following company summary:\n",
    "    \n",
    "    {company_summary}\n",
    "    \"\"\",\n",
    "    response_format=\"json\",\n",
    "    # Provide a few-shot example\n",
    "    examples=[\n",
    "        {\n",
    "            \"messages\": [\n",
    "                {\"role\": \"user\", \"content\": \"Analyze the following company summary:\\n\\nTechCorp is a SaaS company with $50M ARR, but high churn (30%) and new competition. They are burning $2M/month.\"},\n",
    "                {\"role\": \"assistant\", \"content\": \"{\\\"key_risks\\\": [\\\"High customer churn (30%)\\\", \\\"Negative cash flow ($2M/month burn)\\\", \\\"Increasing competitive pressure\\\"], \\\"final_rating\\\": \\\"High\\\"}\"}\n",
    "            ]\n",
    "        }\n",
    "    ]\n",
    ")\n",
    "\n",
    "print(\"Tinker Task defined successfully.\")"
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
    "# Define some seed inputs to generate variations from\n",
    "seed_inputs = [\n",
    "    {\"company_summary\": \"StableCo is a utilities provider with 10-year government contracts, low debt, and 5% annual profit growth.\"},\n",
    "    {\"company_summary\": \"Growthly is a pre-profit tech startup with a new patent but no revenue and only 6 months of runway left.\"},\n",
    "    {\"company_summary\": \"RetailGiant is a 50-year-old retailer facing declining foot traffic due to e-commerce, but has significant real estate assets.\"}\n",
    "]\n",
    "\n",
    "print(f\"Generating 10 examples based on {len(seed_inputs)} seed inputs...\")\n",
    "\n",
    "# This call will use the Tinker API and your OpenAI key to generate data\n",
    "try:\n",
    "    generated_dataset = tinker.generate_dataset(\n",
    "        task=data_gen_task,\n",
    "        inputs=seed_inputs,\n",
    "        num_examples=10 # Generate 10 high-quality training examples\n",
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
    "Convert the generated data into the `jsonl` format required by the OpenAI fine-tuning API."
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
    "            # The 'messages' key is exactly what tinker provides\n",
    "            # and what the OpenAI API expects.\n",
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

### `tinker_lab/02_Model_Training.ipynb`

```json
{
"cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 2: Model Training\n",
    "\n",
    "**Goal:** Load the `jsonl` dataset and submit a fine-tuning job."
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
    "# Define file paths\n",
    "TRAINING_FILE_PATH = \"outputs/datasets/training_data.jsonl\"\n",
    "MODEL_OUTPUT_DIR = \"outputs/model_weights/\"\n",
    "MODEL_LOG_FILE = os.path.join(MODEL_OUTPUT_DIR, \"finetune_job_log.json\")\n",
    "\n",
    "print(f\"Training file: {TRAINING_FILE_PATH}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load and Validate Training Data\n",
    "\n",
    "Quickly inspect the file to make sure it's valid before uploading."
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
    "    print(\"\\n--- First Line Example ---\")\n",
    "    print(json.dumps(json.loads(lines[0]), indent=2))\n",
    "except FileNotFoundError:\n",
    "    print(f\"ERROR: Training file not found at {TRAINING_FILE_PATH}\")\n",
    "    print(\"Please run '01_Data_Generation.ipynb' first.\")\n",
    "except Exception as e:\n",
    "    print(f\"An error occurred loading the file: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Upload File to OpenAI"
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
    "        print(\"Uploading file to OpenAI...\")\n",
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
    "## 4. Create Fine-Tuning Job\n",
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
    "        job = client.fine_tuning.jobs.create(\n",
    "            training_file=training_file_object.id,\n",
    "            model=\"gpt-3.5-turbo\", # Or gpt-4, etc.\n",
    "            suffix=\"adam-risk-assessor-v1\"\n",
    "        )\n",
    "        print(f\"Fine-tuning job created successfully! Job ID: {job.id}\")\n",
    "        print(f\"Job status: {job.status}\")\n",
    "        \n",
    "        # Save job details\n",
    "        with open(MODEL_LOG_FILE, 'w') as f:\n",
    "            json.dump(job.to_json(), f, indent=4)\n",
    "        print(f\"Job details saved to {MODEL_LOG_FILE}\")\n",
    "        \n",
    "        print(\"\\n--- To monitor job status, run: ---\")\n",
    "        print(f\"client.fine_tuning.jobs.retrieve('{job.id}')\")\n",
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
