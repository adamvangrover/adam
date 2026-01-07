# API Reference

This document provides a high-level reference for the core modules of Adam v23.5.

## Core Engine

### MetaOrchestrator

**Location:** `core/engine/meta_orchestrator.py`

The `MetaOrchestrator` is the central "Brain" of the architecture. It routes user queries to the appropriate execution engine based on complexity and intent.

**Key Methods:**

*   `route_request(query: str, context: Optional[Dict[str, Any]]) -> Any`:
    *   **Description:** Analyzes the query complexity and routes it to the best engine (Deep Dive, Swarm, Code Gen, Red Team, etc.).
    *   **Arguments:**
        *   `query` (str): The user's input request.
        *   `context` (dict, optional): Additional context or session data.
    *   **Returns:** The result of the execution (variable type).

*   `_assess_complexity(query: str, context: Dict[str, Any]) -> str`:
    *   **Description:** Heuristic-based routing logic.
    *   **Returns:** A routing key (e.g., `"DEEP_DIVE"`, `"SWARM"`, `"HIGH"`).

*   `_run_deep_dive_flow(query: str, context: Optional[Dict[str, Any]]) -> Any`:
    *   **Description:** Executes the v23.5 Deep Dive Protocol (Graph-based with fallback).

### NeuroSymbolicPlanner

**Location:** `core/engine/neuro_symbolic_planner.py` (Inferred)

The planner breaks high-level goals into executable graphs using a "Plan-on-Graph" approach.

**Key Methods:**

*   `discover_plan(start_node: str, target_node: str) -> Dict`:
    *   **Description:** Finds a path in the Knowledge Graph and generates a plan.
*   `to_executable_graph(plan_data: Dict) -> CompiledGraph`:
    *   **Description:** Compiles the plan into a LangGraph executable.

---

## Agent System

### AsyncAgentBase

**Location:** `core/system/v22_async/async_agent_base.py`

Abstract base class for asynchronous, message-driven agents in the v22 architecture.

**Key Methods:**

*   `execute(*args: Any, **kwargs: Any) -> Any`:
    *   **Description:** Abstract method containing the agent's core logic. Must be implemented by subclasses.
*   `start_listening()`:
    *   **Description:** Subscribes the agent to its message broker topic.
*   `send_message(target_agent: str, message: Dict[str, Any])`:
    *   **Description:** Asynchronously sends a message to another agent.

### AgentOrchestrator

**Location:** `core/system/agent_orchestrator.py` (Inferred)

Legacy/Compatibility orchestrator for managing v21/v22 agents.

---

## Data Processing

### UniversalIngestor

**Location:** `core/data_processing/universal_ingestor.py`

Handles the ingestion, scrubbing, and standardization of data artifacts (documents, code, logs).

**Key Methods:**

*   `scan_directory(root_path: str, recursive: bool = True)`:
    *   **Description:** Scans a directory for ingestible content.
*   `process_file(filepath: str)`:
    *   **Description:** Determines file type and processes it into a `GoldStandardArtifact`.
*   `save_to_jsonl(output_path: str)`:
    *   **Description:** Saves all ingested artifacts to a JSONL file.

### GoldStandardScrubber

**Location:** `core/data_processing/universal_ingestor.py`

Static utility class for cleaning and scoring data.

**Key Methods:**

*   `assess_conviction(content: Any, artifact_type: str) -> float`:
    *   **Description:** Calculates a quality score (0.0 - 1.0) for the content.
*   `clean_text(text: str) -> str`:
    *   **Description:** Standardizes whitespace and encoding.
