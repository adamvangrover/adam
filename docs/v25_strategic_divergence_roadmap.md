# v25 Strategic Divergence Roadmap

## Overview
As per the `AGENTS.md` directive, the Adam development pipeline has bifurcated into two distinct paths for version 25.0+:

1.  **Path A (Product / Odyssey)**: The "Odyssey" Chief Risk Officer (CRO) Copilot.
2.  **Path B (Research / Inference Lab)**: High-performance inference and reasoning optimization.

This document tracks the milestones and status of both paths.

## Path A: The Odyssey System (Product)
**Focus:** Reliability, Auditability, Business Logic, FIBO Integration.
**Target Directory:** `core/vertical_risk_agent/`, `core/agents/orchestrators/`

### 1. The Odyssey Unified Knowledge Graph (OUKG)
- [x] **Schema Definition:** FIBO-aligned schema defined in `data/fibo_knowledge_graph_schema.json`.
- [x] **Ingestion Logic:** `UnifiedKnowledgeGraph.ingest_risk_state` implemented.
- [ ] **Full FIBO Mapping:** Complete mapping of all 200+ fields in the Credit Agreement schema.

### 2. The Hub-and-Spoke Architecture
- [x] **Hub Agent:** `OdysseyHubAgent` (Adam v25.5) implemented in `core/agents/orchestrators/odyssey_hub_agent.py`.
- [ ] **Spoke: CreditSentry:** Refactor `SNCRatingAgent` to align with new OUKG interfaces.
- [ ] **Spoke: Market Mayhem:** Integrate `CrisisSimulationGraph` as a dedicated spoke.
- [ ] **Spoke: Deep Dive:** Formalize `FundamentalAnalystAgent` output for graph ingestion.

### 3. Documentation
- [x] **Strategic Whitepaper:** `docs/whitepapers/odyssey_semantic_architecture.md` created.
- [x] **Configuration:** `config/Adam_v25.5_Portable_Config.json` defined.

## Path B: The Inference Lab (Research)
**Focus:** Velocity, Throughput, Math, Triton/CUDA.
**Target Directory:** `experimental/inference_lab/`

### 1. High-Performance Reasoning
- [ ] **Tree of Thoughts:** Optimize `tree_of_thoughts.py` for sub-50ms step times.
- [ ] **Speculative Decoding:** Implement custom draft models for financial text.

### 2. Infrastructure
- [ ] **KV Cache Optimization:** Implement paged attention for long-context documents (Credit Agreements).
