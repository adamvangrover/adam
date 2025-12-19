#!/bin/bash
# scripts/run_adamv23.sh

# Ensure we are in the root
cd "$(dirname "$0")/.."

# Check for build flag
if [ "$1" == "--build-client-sim" ]; then
    echo "Building Client Simulation..."
    export PYTHONPATH=.
    python3 core/simulations/client_simulation_builder.py
    exit 0
fi

# Load System Prompt
if [ -f "config/AWO_System_Prompt.md" ]; then
    SYSTEM_PROMPT=$(cat config/AWO_System_Prompt.md)
else
    echo "Warning: AWO System Prompt file (config/AWO_System_Prompt.md) not found."
    SYSTEM_PROMPT=""
fi

# Execute
# We use python3 -m core.main so that imports work correctly from root
export PYTHONPATH=.
python3 -m core.main --system_prompt "$SYSTEM_PROMPT" "$@"
