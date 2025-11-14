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
