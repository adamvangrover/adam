# AFOS Execution Kernel Specification

## Overview
The Execution Kernel handles the orchestration, scheduling, and resilience of workflows. It is explicitly decoupled from the business logic (which lives in the Policy and Decision kernels).

## Philosophy: Runtimes are Implementation Details
In AFOS, orchestration tools (like LangGraph, Temporal, or Celery) do not *define* the architecture; they *implement* it. The Execution Kernel abstracts the underlying runtime to ensure the operating system survives regardless of shifts in AI or orchestration frameworks.

## Core Responsibilities

1. **Workflow DSL Parsing:** Interpreting high-level workflow definitions.
2. **Execution Planning:** Mapping the DSL into a schedule of tasks.
3. **Task Scheduling:** Dispatching tasks to available compute resources (including LLM agents).
4. **Agent Runtime:** Providing the sandbox and context for agents to operate.
5. **Checkpointing:** Periodically saving state to the Knowledge Kernel (Temporal Memory) to survive process crashes.
6. **Recovery:** Resuming execution from the last valid checkpoint.
7. **Interrupts:** Handling Human-in-the-Loop (HITL) pauses or policy violations.

## Interaction with LLMs
LLMs exist within the Execution Kernel merely as replaceable execution engines used to process unstructured data or perform semantic reasoning. They are constrained by the deterministic bounds of the Policy Kernel.
