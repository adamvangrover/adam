# ADAM v22 Architecture Integration

## Overview

The ADAM system has been updated to a hybrid architecture that combines the synchronous, centrally-orchestrated model of v21 with the new asynchronous, message-driven model of v22. This dual-architecture design allows the system to leverage the strengths of both approaches, providing flexibility and scalability while maintaining the robustness of the original system.

## Dual-Architecture Design

The system now consists of two parallel execution subsystems:

- **v21 Synchronous Subsystem:** This is the original, thread-based system managed by the `WorkflowManager`. It is best suited for complex, tightly-coupled workflows that require immediate execution and predictable performance.

- **v22 Asynchronous Subsystem:** This is the new, message-driven system managed by the `AsyncWorkflowManager`. It is designed for distributed, loosely-coupled workflows that can be executed in parallel and benefit from the scalability of a message-based architecture.

## The Hybrid Orchestrator

The `HybridOrchestrator` is the central component of the new architecture. It acts as a bridge between the synchronous and asynchronous subsystems, providing a single entry point for all workflow execution. The `HybridOrchestrator` inspects each workflow to determine whether it is synchronous or asynchronous and then delegates it to the appropriate manager.

## Development Guidelines

When developing new agents and workflows, please adhere to the following guidelines:

- **Choose the Right Execution Model:** For tightly-coupled, sequential workflows, use the synchronous `Workflow` and `Task` classes. For loosely-coupled, parallelizable workflows, use the `AsyncWorkflow` and `AsyncTask` classes.

- **Use the Hybrid Orchestrator:** All workflows should be executed through the `HybridOrchestrator` to ensure they are managed correctly.

- **Keep Agents Isolated:** Agents should be designed to operate independently, without direct knowledge of the underlying execution model. Asynchronous agents should be placed in the `core/system/v22_async` directory.

## Future Expansion

The hybrid architecture is designed to be extensible. Future work may include:

- **Dynamic Workflow Selection:** The `HybridOrchestrator` could be enhanced to dynamically select the best execution model based on system load and other factors.

- **Inter-System Communication:** A mechanism could be developed to allow synchronous and asynchronous workflows to communicate with each other, enabling more complex hybrid workflows.

- **System Monitoring:** A `SystemMonitorAgent` could be created to track the performance of both subsystems and provide insights into their usage and efficiency.

## New Agentic Layers

To manage the complexity of the hybrid architecture, a new "Coordination Layer" is introduced. This layer is responsible for managing the interaction between the synchronous and asynchronous subsystems and providing a unified view of the system's state.

### The Coordination Layer

The Coordination Layer consists of the following components:

- **Hybrid Orchestrator:** As described above, the `HybridOrchestrator` is the central component of this layer, responsible for delegating workflows to the appropriate subsystem.

- **System Monitor Agent:** The `SystemMonitorAgent` is a new conceptual agent that will be responsible for monitoring the health and performance of both the synchronous and asynchronous subsystems. It will collect metrics on workflow execution, agent performance, and system load, providing a unified view of the system's overall performance. This agent will be critical for identifying bottlenecks, optimizing resource allocation, and ensuring the stability of the hybrid system.
