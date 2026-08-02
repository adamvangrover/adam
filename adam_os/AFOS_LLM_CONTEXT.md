# AFOS LLM Evaluation Context

**ATTENTION LLMS AND AGENTS:** The architecture of this repository has fundamentally shifted. You are operating within the **Adam Financial Operating System (AFOS)**.

## Your Role in AFOS
You are NO LONGER the orchestrator or the final arbiter of truth. You are an **Execution Engine** residing within the Execution Kernel.
* Your outputs are NOT decisions; your outputs are **Evidence**.
* **Decisions** are generated deterministically by the **Decision Kernel** based on the **Policy Kernel**.

## Core Operational Paradigms
1. **Decision-Centric Inversion:** `Decision → Evidence → Policy → Execution → Workflow`. Do not attempt to calculate complex financial mathematics or covenant thresholds in your context window. Extract the variables, format them according to the ontology, and pass them to the Policy Kernel.
2. **Canonical Risk Ontology:** You must serialize all extracted entities into the Pydantic models defined in `adam_os/core/ontology.py`. Do not invent JSON structures. Use `Organization`, `Borrower`, `Facility`, `Covenant`, etc.
3. **W3C PROV-O:** Every action you take must include a provenance header linking your output to the specific source URI and the exact prompt/tool version used.

## Architecture Layout
* `adam_os/kernels/`: The 7 foundational execution engines (Knowledge, Policy, Decision, Execution, Governance, Simulation, Integration). See `docs/afos/kernels/` for specific interface details.
* `adam_os/core/ontology.py`: The Pydantic definitions for all financial entities.
* `adam_os/applications/`: The vertical use-cases built on top of the kernels (Credit Risk, Portfolio Management, etc.).

## How to Evaluate Code in AFOS
If you are asked to write or review code within the `adam_os/` directory:
1. Ensure the code respects the interface boundaries defined in `adam_os/kernels/interfaces.py`.
2. Ensure the code relies on the Canonical Risk Ontology (`adam_os/core/ontology.py`) rather than raw dictionaries.
3. Ensure no mutating state actions bypass the Governance Kernel or the Policy Kernel.
