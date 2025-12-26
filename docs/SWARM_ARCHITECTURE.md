# Swarm Architecture (Hive Mind)

## Overview
The Swarm Architecture allows Adam to scale horizontally by deploying multiple lightweight, specialized agents ("Workers") managed by a central "Hive Mind". This architecture is designed for parallelizable tasks such as:
- Wide-net news scanning.
- Analyzing large portfolios of assets.
- Distributed data gathering.

## Core Components

### 1. PheromoneBoard (`core/engine/swarm/pheromone_board.py`)
A shared "blackboard" that implements the **Stigmergy** pattern. Agents communicate indirectly by depositing "pheromones" (signals) that persist and decay over time.
- **Deposit**: Agents leave a signal (e.g., `TASK_ANALYST`, `RESULT`).
- **Sniff**: Agents look for active signals above a certain intensity.
- **Decay**: Signals fade over time to prevent stale data accumulation.

### 2. SwarmWorker (`core/engine/swarm/worker_node.py`)
The fundamental unit of the swarm.
- **Lifecycle**: `sniff` -> `consume` -> `execute` -> `deposit`.
- **Roles**: Workers can be specialized (e.g., `AnalysisWorker`) or generalist.
- **Autonomy**: Workers pull tasks rather than having them pushed, ensuring load balancing.

### 3. HiveMind (`core/engine/swarm/hive_mind.py`)
The orchestrator that bridges the `MetaOrchestrator` and the Swarm.
- Initializes the worker pool.
- Disperses high-level user requests as pheromone tasks.
- Gathers results from the board.

## Integration with MetaOrchestrator
The `MetaOrchestrator` detects queries containing keywords like "swarm", "scan", or "hive". It routes these to the `HiveMind`, which then spins up the swarm (lazy loading) and executes the parallel workflow.

## Future Roadmap
- **Evolutionary Algorithms**: Allow workers to "breed" (combine successful prompts) to improve performance.
- **Cross-Hive Communication**: Allow multiple Hive Minds to share a global Pheromone Board.
- **Hardware Acceleration**: Run workers on separate threads/processes or even separate containers.
