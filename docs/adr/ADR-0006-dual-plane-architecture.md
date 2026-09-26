# ADR-0006-Dual-Plane-Architecture

## Context
Regulatory pressures (Basel III/IV, FRTB, SR 11-7) mandate strict separation of execution (1L) and independent risk governance (2L). Legacy monolithic risk stacks suffer from tight coupling, where governance checks introduce unacceptable latency to front-office execution.

## Decision
Implement a Dual-Plane Architecture, physically and logically segregating the 1L Data Plane (execution) from the 2L Control Plane (governance).
- **1L Data Plane:** Optimized for sub-millisecond execution latency using deterministic models. Emits high-throughput, asynchronous telemetry.
- **2L Control Plane:** Operates asynchronously (near real-time). Ingests 1L telemetry to run rigorous independent challenger models, enforce jsonLogic gating rules, and manage W3C PROV-O compliance logging.
- **Transport:** Decentralized event brokerage via Apache Kafka.

## Status
Accepted

## Consequences
- **Latency Isolation:** Front-office critical paths are unblocked.
- **Compliance:** Enforces true segregation of duties and model risk management.
- **Complexity:** Requires robust, zero-loss telemetry extraction and CI/CD organizational shifts between QFO and MRM teams.
