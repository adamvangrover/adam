# System 1: The Async Swarm

The `core/system/v22_async/` directory implements the **System 1** component of Adam v26.0.

## ⚡ What is System 1?

System 1 is the "Fast, Intuitive, and Automatic" mode of thinking (Daniel Kahneman). In Adam, this corresponds to the **Async Swarm**—a collection of lightweight agents that perform non-blocking tasks in parallel.

## 🏗️ Architecture

The Swarm uses an **Event-Driven Architecture** (EDA) to handle high throughput.

### Key Components

*   **`AsyncAgentBase` (`async_agent_base.py`)**: The base class for all Swarm Agents. It handles:
    *   Message queue subscription.
    *   Task lifecycle management (Start -> Running -> Complete/Fail).
    *   Error logging.

*   **`AsyncTask` (`async_task.py`)**: A data object representing a unit of work.
    *   Attributes: `task_id`, `priority`, `payload`, `deadline`.

*   **`AsyncWorkflowManager` (`async_workflow_manager.py`)**: The conductor that assigns tasks to available workers based on load and capability.

## 🚀 When to use System 1?

Use the Swarm for tasks that are:
1.  **Independent:** Fetching news for 50 different stocks.
2.  **Latency-Sensitive:** Real-time sentiment scoring of Twitter feeds.
3.  **IO-Bound:** Scraping websites or querying external APIs.

For complex, multi-step reasoning, use **System 2** (`core/engine/`).
