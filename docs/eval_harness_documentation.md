# Unified Evaluation Harness Documentation

## Overview

The Unified Evaluation Harness consolidates the execution of multiple independent evaluation suites within the `evals/` directory into a single, cohesive test run. It generates a machine-readable JSON report and powers a human-readable interactive HTML dashboard to visualize the results of the agentic systems and models.

## Architecture

The harness is composed of two primary components:

1.  **`evals/unified_eval.py` (Execution Engine)**
    *   Acts as the orchestrator.
    *   Discovers and sequentially runs targeted evaluation scripts (e.g., `eval_crisis_sim.py`, `eval_rag_pipeline.py`, `run.py`).
    *   Captures `stdout`, `stderr`, execution duration, and exit status codes for each test suite.
    *   Aggregates the data and writes a structured JSON report to `evals/data/unified_eval_report.json`.

2.  **`showcase/adam_eval_dashboard.html` (Interactive User Interface)**
    *   A standalone, browser-based dashboard.
    *   Dynamically fetches the latest `unified_eval_report.json`.
    *   Parses the machine-readable JSON into human-readable cards displaying the status (Pass/Fail), execution time, and detailed logs for each evaluation suite.
    *   Provides a "Refresh Data" mechanism to pull the latest test results without reloading the entire application.

## How to Use

### 1. Running the Evaluations (Machine Execution)

To generate a new evaluation report, run the unified script from the root of the repository:

```bash
PYTHONPATH=. uv run python evals/unified_eval.py
```

This will execute the sub-evaluations and create/update the `evals/data/unified_eval_report.json` file.

### 2. Viewing the Results (Human Review)

To view the results interactively:

1.  Start a local HTTP server from the repository root (e.g., `python -m http.server 8000`).
2.  Navigate to `http://localhost:8000/showcase/adam_eval_dashboard.html` in your web browser.
3.  The dashboard will automatically load the latest JSON report. If you re-run the evaluations, simply click the "Refresh Data" button on the dashboard to see the updated results.

## Extending the Harness

To add a new evaluation suite to the unified harness:

1.  Create your evaluation script within the `evals/` directory (e.g., `evals/my_new_test.py`). Ensure it returns an appropriate exit code (0 for success, non-zero for failure).
2.  Open `evals/unified_eval.py`.
3.  Add a new `run_command` block targeting your script within the `main()` function.
4.  Append the results to the `results["evaluations"]` dictionary using a descriptive key.

The new evaluation will automatically appear on the interactive dashboard upon the next execution.
