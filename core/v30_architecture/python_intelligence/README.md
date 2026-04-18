# Python Intelligence (v30 Architecture)

This directory contains the modernized Python-based multi-agent system (`v30`), built around a Swarm/Mesh architecture.

## Overview

The `v30` architecture focuses on distributed, asynchronous intelligence. Agents communicate via a unified `NeuralMesh` using standardized `emit_packet` workflows.

### BaseAgent Implementations

All operational agents inherit from `BaseAgent` (`core/v30_architecture/python_intelligence/agents/base_agent.py`), which provides standardized telemetry and NeuralMesh integration.

**Usage Example:**
```python
from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="CustomAgent01", role="Specialized Analyzer")

    async def execute_task(self, data):
        # ... logic ...
        await self.emit(packet_type="THOUGHT", payload={"status": "processing"})
```

### NeuralMesh Interactions

The `NeuralMesh` (`core/v30_architecture/python_intelligence/bridge/neural_mesh.py`) handles real-time broadcasting and inter-agent communication via WebSockets.

**Usage Example:**
```python
from core.v30_architecture.python_intelligence.bridge.neural_mesh import NeuralPacket, emit_packet

packet = NeuralPacket(
    source_agent="RiskGuardian",
    packet_type="risk_alert",
    payload={"level": "high", "asset": "TSLA"}
)
await emit_packet(packet)
```

### emit_packet Workflows

The `emit_packet` function is the primary entry point for sending data into the NeuralMesh. It asynchronously dispatches `NeuralPacket` objects to all subscribed clients.
