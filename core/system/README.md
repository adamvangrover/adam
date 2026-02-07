# Adam System 1: The Swarm Infrastructure

The `core/system/` directory contains the low-level infrastructure that powers the "System 1" (Fast) cognitive capabilities of Adam. It handles asynchronous execution, event routing, and memory management.

## 🐝 The Swarm Architecture

Unlike the "System 2" Planner which is deliberate and sequential, the Swarm is designed for **speed, concurrency, and reactivity**.

### Key Components

*   **`v22_async/`**: The heart of the Swarm.
    *   **`AsyncAgentBase`**: The base class for all lightweight workers. It supports non-blocking I/O and fire-and-forget patterns.
    *   **Event Loop**: A custom asyncio loop that manages thousands of concurrent micro-tasks (e.g., fetching 50 news feeds simultaneously).
*   **`message_broker.py`**: The nervous system.
    *   Implements a Pub/Sub mechanism allowing agents to subscribe to topics (e.g., `market.news.aapl`) without knowing who publishes them.
*   **`memory_manager.py`**: The hippocampus.
    *   Manages short-term context windows and commits important facts to the long-term Vector DB.

## 🚀 Usage Pattern

System 1 is typically used for:
1.  **Data Ingestion:** Scrapers and API pollers.
2.  **Monitoring:** "Watchdogs" that trigger alerts on specific conditions (e.g., "Price < $150").
3.  **Pre-processing:** Cleaning and normalizing data before handing it to System 2.

### Example: Async Agent

```python
from core.system.v22_async.agent import AsyncAgentBase

class NewsWatcher(AsyncAgentBase):
    async def run(self):
        while True:
            news = await self.fetch_latest()
            if "crisis" in news.title:
                # Fire and forget - don't wait for a response
                await self.publish("risk.alert", news)
```

## ⚠️ Comparison with System 2

| Feature | System 1 (Swarm) | System 2 (Engine) |
| :--- | :--- | :--- |
| **Speed** | Milliseconds | Seconds/Minutes |
| **Logic** | Heuristic / Reflexive | Reasoning / Deliberate |
| **State** | Stateless (mostly) | Stateful (Graph) |
| **Use Case** | Monitoring, Fetching | Analysis, Planning |
