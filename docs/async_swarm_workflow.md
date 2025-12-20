# Async Swarm Workflow Documentation

## Overview

The Async Swarm Workflow represents the v22 architecture shift in the ADAM system, moving from a purely synchronous, orchestrator-led model to an asynchronous, message-driven "swarm" of agents. This architecture enables higher concurrency, improved resilience, and more complex emergent behaviors.

## Key Components

### 1. Message Broker (`core/system/message_broker.py`)
The central nervous system of the swarm. It handles:
- **Asynchronous Messaging:** Agents publish and subscribe to topics.
- **Event-Driven Execution:** Tasks are triggered by events rather than direct function calls.
- **Decoupling:** Agents do not need to know about each other's existence, only the message schemas.

### 2. Async Agent Base (`core/system/v22_async/async_agent_base.py`)
The base class for all swarm agents. Features include:
- **Non-blocking I/O:** Built on `asyncio` for efficient resource usage.
- **State Management:** Local state handling independent of the global orchestrator.
- **Lifecycle Management:** Start, stop, and pause capabilities.

### 3. Hybrid Orchestrator (`core/system/hybrid_orchestrator.py`)
Bridges the v21 synchronous world with the v22 swarm. It:
- **Routes Tasks:** Decides whether a task requires a linear plan (v21) or a swarm solution (v22).
- **Aggregates Results:** Collects outputs from asynchronous agents for final reporting.

## Workflow Lifecycle

1.  **Task Ingestion:** A high-level goal is received by the `MetaOrchestrator`.
2.  **Decomposition:** If suitable for the swarm, the task is broken down into independent sub-tasks.
3.  **Broadcast:** Sub-tasks are published to the message broker.
4.  **Swarm Activation:** Idle agents subscribed to relevant topics pick up tasks.
5.  **Execution & Collaboration:** Agents execute tasks, potentially generating new sub-tasks or requesting help via the broker.
6.  **Convergence:** Results are published back to a result topic.
7.  **Synthesis:** The `HybridOrchestrator` gathers results and formulates the final response.

## Benefits

- **Scalability:** Add more agent instances to handle increased load without architectural changes.
- **Robustness:** Failure of a single agent does not halt the entire process; the task can be retried or picked up by another agent.
- **Real-Time Responsiveness:** The system can react to incoming data streams (e.g., market ticks) immediately.

## Developer Guide

To create a new async agent:
1.  Inherit from `AsyncAgentBase`.
2.  Implement the `process_message` method.
3.  Register the agent with the `MessageBroker` and subscribe to topics.

```python
class MySwarmAgent(AsyncAgentBase):
    async def process_message(self, message: Message):
        # Process logic
        result = await self.do_work(message.payload)
        await self.publish("result_topic", result)
```
