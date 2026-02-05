# Adam v26.0 Setup Guide

This document provides detailed instructions for installing Adam on various platforms and configuring it for production.

## Table of Contents
*   [System Requirements](#system-requirements)
*   [Local Development Setup](#local-development-setup)
    *   [Linux/macOS](#linuxmacos)
    *   [Windows](#windows)
*   [Docker Deployment](#docker-deployment)
*   [Troubleshooting](#troubleshooting)

## System Requirements

*   **OS:** Linux (Ubuntu 22.04+ recommended), macOS (Ventura+), or Windows 11 (WSL2).
*   **Memory:** 16GB RAM minimum (32GB recommended for running local LLMs).
*   **Storage:** 10GB free space.
*   **Python:** Version 3.10, 3.11, or 3.12.

## Local Development Setup

### Linux/macOS

1.  **Install System Dependencies:**
    ```bash
    # Ubuntu
    sudo apt update && sudo apt install git curl build-essential libpq-dev

    # macOS (Homebrew)
    brew install git curl postgresql
    ```

2.  **Install uv (Package Manager):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Clone and Sync:**
    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    uv sync
    ```

4.  **Activate Environment:**
    ```bash
    source .venv/bin/activate
    ```

### Windows

We strongly recommend using **WSL2 (Windows Subsystem for Linux)** for the best experience. If you must use native Windows:

1.  **Install Python:** Download and install Python 3.10+ from python.org.
2.  **Install uv:**
    ```powershell
    pip install uv
    ```
3.  **Clone and Sync:**
    ```powershell
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    uv sync
    ```
4.  **Activate Environment:**
    ```powershell
    .venv\Scripts\activate
    ```

## Docker Deployment

Adam is container-ready.

1.  **Build the Image:**
    ```bash
    docker build -t adam-v26 .
    ```

2.  **Run the Container:**
    ```bash
    docker run -d -p 5000:5000 --env-file .env adam-v26
    ```

    *Note: Ensure your `.env` file is properly configured before running.*

## Troubleshooting

### "Command not found: uv"
Ensure `~/.cargo/bin` or the installation directory is in your PATH. Try restarting your shell.

### "Missing dependency: docling"
Some libraries require system-level tools. On Linux, try `sudo apt install libmagic1`.

### "OpenAI API Error"
Check your `.env` file. Ensure `OPENAI_API_KEY` is set and has credits.
