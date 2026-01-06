# Meta-Agents Development Guide

## Overview

The Adam system has been expanded with a triad of "Meta-Agents" designed to operate at a higher level of abstraction than standard task-oriented agents. These agents focus on system evolution, knowledge transfer, and temporal context.

## 1. Evolutionary Architect Agent (`core/agents/meta_agents/evolutionary_architect_agent.py`)

**Purpose:** To drive the codebase forward through autonomous code evolution ("Active Inference").

**Key Features:**
- **Action-Oriented:** Predisposed to propose and draft changes rather than just analyze.
- **Safety-First:** Implements AST (Abstract Syntax Tree) parsing to ensure proposed Python code is syntactically valid before presentation.
- **Hybrid Generation:** Designed to use the Semantic Kernel for real code generation when available, falling back to structured mock logic for testing/bootstrapping.

**Extension Points:**
- **Refinement Node:** Implement a feedback loop where the agent iterates on the code based on linter output (e.g., `pylint` or `mypy`).
- **Git Integration:** Give the agent the ability to create actual Git branches and commits via a `GitPlugin`.

## 2. Didactic Architect Agent (`core/agents/meta_agents/didactic_architect_agent.py`)

**Purpose:** To bridge the gap between code and comprehension by generating tutorials and portable setups.

**Key Features:**
- **Real Context:** Reads actual source code files to generate accurate snippets and explanations.
- **Portable Configs:** automatically generates `Dockerfile` and `requirements.txt` to make the tutorial runnable in isolation.
- **Template-Based:** Uses Jinja2 (if available) for flexible content generation.

**Extension Points:**
- **Interactive Notebooks:** Extend to generate `.ipynb` Jupyter Notebooks instead of just Markdown.
- **Video Scripting:** Generate scripts for automated video tutorials.

## 3. Chronos Agent (`core/agents/meta_agents/chronos_agent.py`)

**Purpose:** To manage temporal state and find historical analogs.

**Key Features:**
- **Multi-Horizon Memory:** Manages Short, Medium, and Long-term memory buckets.
- **Archival Scanning:** Scans the `core/libraries_and_archives/reports/` directory to ground long-term memory in actual system artifacts.
- **Temporal Synthesis:** Synthesizes a narrative connecting current context with historical patterns.

**Extension Points:**
- **Vector Database:** Connect "Long Term" memory to a Vector DB (e.g., Pinecone, Milvus) for semantic search over millions of documents.
- **Git History:** Integrate with the Git log to understand code evolution over time (e.g., "Who changed this line last year and why?").

## Development Philosophy

These agents follow the **"Additive"** principle: they should be able to run alongside existing system components without modifying core logic. They are designed to be "overlay" intelligence that observes and suggests, rather than blocking the critical path.

### Testing
Use `tests/verify_meta_agents.py` to verify basic functionality. Ensure `numpy` and `pydantic` are installed in your environment.
