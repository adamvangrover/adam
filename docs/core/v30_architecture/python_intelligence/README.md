# Python Intelligence (v30 Architecture)

This directory (`core/v30_architecture/python_intelligence/`) contains the foundational components for the modern, modular multi-agent system.

## Key Components

### `BaseAgent` (`agents/base_agent.py`)
The `BaseAgent` class serves as the core foundation for all AI agents in the v30 architecture.
- All new agents must inherit from `BaseAgent`.
- Each agent must define a `name` and a `role`.
- Agents communicate primarily via the asynchronous `emit` method, which standardizes telemetry and inter-agent communication.
- The `execute(**kwargs)` method should be implemented by subclasses to support thread safety and async compatibility.

### `NeuralMesh` (`bridge/neural_mesh.py`)
The `NeuralMesh` is a high-speed websocket-based event bus that facilitates communication across the swarm.
- It acts as the backbone for inter-agent packet routing.
- Designed to handle real-time broadcasts and directed telemetry without traditional synchronous blocking.

### `emit_packet` Workflow
Agents push data into the `NeuralMesh` using the `emit_packet` function.
- **Workflow**: `Agent.emit() -> emit_packet(NeuralPacket) -> NeuralMesh.broadcast() -> Listening Clients/Dashboards`
- A `NeuralPacket` includes the `source_agent`, `packet_type`, and a robust `payload` dictionary.
- This ensures all thoughts, actions, and decisions are perfectly logged and observable by UI dashboards.

## Example Usage

```python
from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Analyzer-1", role="Market Analyst")

    async def execute(self, **kwargs):
        # Perform analysis...
        result = {"status": "complete", "finding": "bullish"}

        # Emit findings to the mesh
        await self.emit(packet_type="ANALYSIS_COMPLETE", payload=result)
```
