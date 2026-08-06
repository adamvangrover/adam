# ADR 0002: Implementation of Event Sourcing Core (CQRS)

## Status
Accepted

## Context
ADAM OS acts as a financial operating system. One of the core requirements outlined in the permanent system prompt is that financial calculations and state transitions must be Deterministic, Auditable, Replayable, Explainable, and Versioned. Traditional CRUD (Create, Read, Update, Delete) databases overwrite historical state, meaning the "why" and "when" of a state change is lost unless explicitly modeled with audit logs (which are often brittle and separate from the source of truth). We need a mechanism to never overwrite financial history and support temporal debugging.

## Decision
We will implement an Event Sourcing Core using the CQRS (Command Query Responsibility Segregation) pattern.
1. We will represent all financial state transitions as immutable events (e.g., `FinancialEvent`).
2. We will build an append-only `EventLedger` to store these events.
3. Current state will not be persisted directly as the source of truth; instead, the source of truth is the event log. State is reconstructed by replaying these events through a reducer function (aggregate root).
4. Each event will have a deterministically computed hash upon creation to ensure cryptographic immutability and detect tampering.

## Alternatives Considered
*   **Traditional CRUD SQL Database with History Tables:** Rejected. History tables can get out of sync with the main table, and relying on database triggers for auditability obscures business logic and makes replays difficult inside application code.
*   **Blockchain / Distributed Ledger (e.g., Hyperledger):** Rejected. Overkill for the current architecture. We need the conceptual benefits of a ledger (immutability, replayability) without the overhead of consensus mechanisms and decentralized nodes.

## Tradeoffs & Consequences
*   **Tradeoff:** Event Sourcing introduces significant conceptual and architectural complexity. Queries become harder because current state isn't explicitly stored (requiring Read Models/Projections).
*   **Tradeoff:** Data storage requirements will grow monotonically over time since data is never deleted.
*   **Consequence:** We achieve perfect auditability and replayability. Temporal workflows can deterministically reconstruct the state of a financial entity exactly as it was at any point in time, enabling highly accurate simulations and backtesting.

## Future Extensions
*   Replace the in-memory `EventLedger` with a durable, scalable event store like EventStoreDB, Apache Kafka, or a specialized append-only PostgreSQL schema.
*   Implement dedicated "Read Models" (Projections) that subscribe to the event stream and update optimized views for querying.
*   Implement event snapshotting to optimize the replay process for long-lived aggregates.
