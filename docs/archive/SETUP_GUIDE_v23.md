# Setup Guide

This guide describes how to set up the Adam v23.5 Financial Intelligence System.

## Prerequisites

- Python 3.10+
- `uv` or `pip` for package management

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    Using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

    Or using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Copy `.env.example` to `.env` and configure your API keys (e.g., OpenAI, Anthropic, Financial Data Providers).

## Running the System

### MCP Server
To run the Model Context Protocol (MCP) server:
```bash
python server/server.py
```

### Running Tests
To ensure the system is working correctly:
```bash
pytest
```

## Troubleshooting

- **Import Errors:** Ensure `PYTHONPATH` includes the root directory.
  ```bash
  export PYTHONPATH=$PYTHONPATH:.
  ```
