# AFOS
Autonomous Financial Operating System - Credit Risk & Underwriting Engine

## Overview
AFOS is a self-contained modular sub-directory application within the repository.
It serves as a Credit Risk & Underwriting Engine, built as a vertically integrated slice.

## Development Setup

The project uses `uv` for dependency management and `hatchling` as the build backend.

### Running Tests

To run the test suite:
```bash
cd afos
PYTHONPATH=$(pwd) uv run pytest
```