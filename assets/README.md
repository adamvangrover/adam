# Adam Assets

## Overview
This module (`assets`) operates as the central repository explicitly meant to store text prompts, heuristic JSON rules, and document artifacts separated from code execution loops.

## Objectives
- Extract hardcoded strings out of `.py` execution files.
- Resolve LLM context window saturation by ensuring execution code strictly reads prompts via dynamic utility loaders instead of embedding large text blocks.

## Future Plans
- Aggressively extract the remaining 50+ specialized heuristic prompts currently inside `core/agents/` into `.md` and `.json` formats within this directory.
- Architect and stabilize a unified loader utility for all swarm agents to draw out structured instructions seamlessly at runtime.
