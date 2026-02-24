# Adam v26.0 User Guide

This guide provides comprehensive instructions on how to use Adam v26.0, the Institutional-Grade Neuro-Symbolic Financial Sovereign.

## 📋 Table of Contents
*   [Running the System](#running-the-system)
*   [CLI Modes](#cli-modes)
*   [Knowledge Graph](#knowledge-graph)
*   [API Usage](#api-usage)
*   [Analysis Modules](#analysis-modules)

---

## Running the System

The primary entry point for Adam is the `scripts/run_adam.py` CLI utility.

### Basic Usage

To launch the system in its default mode:

```bash
python scripts/run_adam.py
```

### CLI Options

Adam supports several command-line arguments to tailor execution:

```bash
usage: run_adam.py [-h] [--query QUERY] [--system_prompt SYSTEM_PROMPT]
                   [--system_prompt_path SYSTEM_PROMPT_PATH] [--legacy]

Adam v26.0 Execution

options:
  -h, --help            show this help message and exit
  --query QUERY         Single query to execute (e.g., "Analyze AAPL credit risk")
  --system_prompt SYSTEM_PROMPT
                        System Prompt to inject (String)
  --system_prompt_path SYSTEM_PROMPT_PATH
                        System Prompt to inject (File Path)
  --legacy              Force usage of legacy v23 graph engine components
```

### Examples

**1. Run a Specific Query:**
```bash
python scripts/run_adam.py --query "Perform a deep dive analysis on Tesla (TSLA)"
```

**2. Use a Custom System Prompt:**
```bash
python scripts/run_adam.py --query "Evaluate geometric growth" --system_prompt_path prompt_library/math_expert.md
```

**3. Run in Legacy Mode (v23 Engine):**
```bash
python scripts/run_adam.py --legacy
```

---

## Knowledge Graph

Adam v26.0's knowledge graph is a rich repository of financial concepts, models, and data, organized in a structured and interconnected manner. It enables Adam to perform in-depth analysis, provide context-aware insights, and generate actionable recommendations.

### Accessing the Knowledge Graph

*   **API:** The Adam API provides endpoints for retrieving and updating information.
*   **Direct Access:** `core/system/knowledge_base.py` manages graph interactions.

---

## API Usage

The Adam API provides a unified interface for interacting with the system.

### API Documentation

Detailed API documentation is available in `docs/api_docs.yaml`.

---

## Analysis Modules

Adam v26.0 provides various analysis modules that can be used to gain insights into financial markets and make informed investment decisions.

*   **Risk Assessment:** Calculates PD, LGD, and Recovery Rates.
*   **Fundamental Analysis:** DCF, LBO, and 3-Statement Modeling.
*   **Market Sentiment:** Real-time news and social sentiment scoring.
*   **Macro Analysis:** Global liquidity and regime classification.
