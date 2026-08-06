# AFOS Knowledge Kernel Specification

## Overview
The Knowledge Kernel serves as the institutional memory for the Adam Financial Operating System (AFOS). It moves beyond simple vector search, implementing a rigorous three-tier memory architecture to answer fundamentally different epistemological questions.

## Memory Tiers

### 1. Transactional Memory
* **Technology:** PostgreSQL
* **Purpose:** The source of truth for structured, immutable facts and ledgers.
* **Question Answered:** "What is true?"
* **Examples:** Current loan balances, covenant threshold values, facility limits.

### 2. Semantic Memory
* **Technology:** Qdrant (Vector Database)
* **Purpose:** High-dimensional embedding storage for unstructured text and conceptual relationships.
* **Question Answered:** "What is similar?"
* **Examples:** Matching current earnings call sentiment to past distress events; finding peer companies with similar risk profiles.

### 3. Relational Memory
* **Technology:** Knowledge Graph (Neo4j/NetworkX)
* **Purpose:** Topological mapping of entities, corporate hierarchies, and dependencies.
* **Question Answered:** "What is connected?"
* **Examples:** Identifying Ultimate Parent entities, mapping cross-default clauses, analyzing sponsor exposure across the portfolio.

### 4. Temporal Memory
* **Technology:** Event Store (Kafka/CQRS)
* **Purpose:** Immutable append-only log of all system state changes.
* **Question Answered:** "What happened?"
* **Examples:** Reconstructing the exact sequence of events leading to a covenant breach; audit trails for regulatory compliance.

## Integration
The Knowledge Kernel is abstracted behind the `IKnowledgeKernel` interface in `adam_os/kernels/interfaces.py`, ensuring that reasoning and execution agents interact with memory without needing to manage the underlying database connections.
