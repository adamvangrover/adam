# Adam Architecture Master Plan

This repository strictly separates System 1 probabilistic swarms, System 2 deterministic reasoners, and purely deterministic financial execution to address the LLM attention decay.

## Modules:
1. `adam_finance`: Houses deterministic math algorithms (VaR, SNC calculations).
2. `adam_interfaces`: Dependency injection interface definitions.
3. `adam_swarm`: Neural swarm and asynchronous high-velocity data processing.
4. `adam_graph`: Neuro-symbolic workflow logic.
5. `adam_governance`: API gatekeeping, state machine, logging.
6. `assets`: Central repository for prompt evictions to prevent context window saturation.

## Sub-Plans
- `adam_finance`: Expand deterministic models; add Monte Carlo pipelines.
- `adam_swarm`: Refine multi-agent orchestration for asynchronous ingestion.
- `assets`: Continually move hardcoded logic strings out of `.py` execution files.

## Review and Merge
The goal is an enterprise-grade functional structure aligned with Diátaxis and AI memory scaling.

## Expansive Sub-Plans

### 1. `adam_finance/` (Deterministic Math Execution)
- **Objective:** Host all pure financial mathematical functions and logic algorithms.
- **Future Steps:** Expand `math.py` to cover advanced Greek calculations for options. Move out more modules that are currently mixed in `core/engine/` over time.
- **Dependencies:** This module should remain free of any dependency on `langchain`, `semantic_kernel`, or other LLM execution tools.

### 2. `adam_interfaces/` (Strict Type Enforcement)
- **Objective:** Define all inter-module API boundaries using `Protocol` and `ABC`.
- **Future Steps:** Introduce domain boundaries to restrict `adam_swarm` output directly passing into `adam_graph` without interface transformation layers.

### 3. `adam_swarm/` (System 1: Fast Heuristics)
- **Objective:** Collect multi-agent orchestrators designed for reading raw earnings calls and parsing text.
- **Future Steps:** Introduce specific API boundary tools mapped to FastMCP schemas dynamically.

### 4. `assets/` (Knowledge & Prompt Separation)
- **Objective:** Continue aggressive prompt eviction. Large hardcoded prompts create context saturation in language models that read Python files.
- **Future Steps:** Extract the rest of the 50+ specialized prompts in `core/agents/` to `.md` or `.json` formats. Maintain a unified loader utility for agents to pull their instructions at runtime.

### 5. `adam_governance/` (State Control)
- **Objective:** Act as the deterministic gatekeeper that forces swarm agents to prove mathematically sound answers before advancing.
- **Future Steps:** Enhance `ProofOfThoughtLogger` to enforce W3C PROV-O compliance strictly on every LLM-generated JSON payload.
