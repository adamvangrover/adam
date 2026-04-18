# Daily Ritual Documentation

The `scripts/daily_ritual.py` script serves as the automated recursive execution wrapper for Protocol ARCHITECT_INFINITE. It acts as the heartbeat of the daily intelligence generation cycle.

## Protocol ARCHITECT_INFINITE
Protocol ARCHITECT_INFINITE dictates a purely additive biological growth model for the system in four phases: Audit, Harvest, Build, Memory. The `daily_ritual.py` script is responsible for initiating this sequence autonomously.

## Parsing Logic and LLM Fallback
The script interfaces with large language models via `litellm` to dynamically generate tasks or code components. It utilizes a robust regex-based parser to extract actionable blocks from the LLM's raw markdown output.

### The `MOCK_PAYLOAD` Fallback
If the live LLM environment is unavailable (e.g., missing API keys or network failure), the script degrades gracefully by utilizing a predefined `MOCK_PAYLOAD`.
- The `MOCK_PAYLOAD` contains a functional, hardcoded output block representing a complete agent response.
- This ensures the daily pipeline can be tested, demonstrated, and continuously integrated in static or isolated environments without relying on external API calls.
