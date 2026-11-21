#!/bin/bash
# Script to run stable tests for Adam v23.5

echo "Running Stable Tests..."
export PYTHONPATH=.

# Core Agent Tests
python -m pytest tests/unit/test_agents.py
python -m pytest tests/unit/test_agent_orchestrator.py
python -m pytest tests/unit/test_code_alchemist.py

# Add more as they are stabilized
echo "Done."
