# Scripts

This directory contains scripts for automating various tasks in the ADAM system. These scripts can be used to run simulations, process data, generate reports, and perform other common tasks.

## Scripting Examples

Here are some examples of how to use the most common scripts in this directory:

### `run_adam.py`

This script is the main entry point for the ADAM system. It starts the system and loads all of the configured agents.

```bash
python scripts/run_adam.py
```

### `run_simulations.sh`

This script runs a suite of simulations to test and evaluate the performance of the ADAM system. You can specify which simulations to run and how many times to run them.

```bash
./scripts/run_simulations.sh --simulation Credit_Rating_Assessment_Simulation --iterations 10
```

### `generate_report.py`

This script generates a variety of reports, such as a daily market briefing or a weekly portfolio summary. You can specify the type of report to generate and the output format.

```bash
python scripts/generate_report.py --report-type daily_briefing --output-format pdf
```

## Available Scripts

*   **`daily_headlines.py`:** Generates a daily news headlines report.
*   **`data_processing.py`:** Processes raw data and prepares it for analysis.
*   **`extract_xai_reasoning.py`:** Extracts explanations from the XAI (Explainable AI) models.
*   **`generate_newsletter.py`:** Generates a weekly newsletter.
*   **`main.py`:** The main entry point for the ADAM system.
*   **`rag_agent_example.py`:** An example of how to use the RAG (Retrieval-Augmented Generation) agent.
*   **`report_generation.py`:** Generates a variety of reports.
*   **`run_adam.py`:** Runs the ADAM system.
*   **`run_simple_simulation.py`:** Runs a simple simulation.
*   **`run_simulations.sh`:** Runs a suite of simulations.
*   **`setup_agent.py`:** Sets up a new agent.

## Running a Script

To run a script, you can use the `python` interpreter. For example, to run the `daily_headlines.py` script, you would use the following command:

```bash
python scripts/daily_headlines.py
```

Some scripts may require command-line arguments. For more information on how to use a specific script, please refer to the documentation within the script itself.

## Creating a New Script

When creating a new script, please follow these guidelines:

*   **Be well-documented.** Include a docstring at the beginning of the script that explains what the script does and how to use it.
*   **Be modular.** Break your script down into smaller, reusable functions.
*   **Use command-line arguments.** Use the `argparse` module to create a command-line interface for your script.
*   **Be idempotent.** Whenever possible, make your scripts idempotent, meaning that they can be run multiple times without changing the result.

By following these guidelines, you can help to ensure that the scripts in the ADAM system are easy to use, maintain, and extend.
