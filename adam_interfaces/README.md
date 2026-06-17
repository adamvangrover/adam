# Adam Interfaces Module

## Overview
This module strictly defines the inter-module contracts between the system 1 swarm and the system 2 graph, separating execution layers from probabilistic layers.

## Structure
- `interfaces.py`: Python Protocols and Abstract Base Classes (ABCs) representing required inputs and methods across components.

## Architectural Principles
- **Contracts:** Codebase boundary interfaces prevent LLM hallucinations by explicit dependencies.
